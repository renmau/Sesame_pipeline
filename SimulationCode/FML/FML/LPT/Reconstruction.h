#ifndef RECONSTRUCTION_HEADER
#define RECONSTRUCTION_HEADER
#include <array>
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Smoothing/SmoothingFourier.h>

namespace FML {
    namespace COSMOLOGY {
        namespace LPT {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;
            template <class T>
            using MPIParticles = FML::PARTICLE::MPIParticles<T>;
            using FloatType = FML::GRID::FloatType;

            //============================================================================
            ///
            /// RSD removal for particles in a periodic box
            /// We use a fixed line of sight direction los_direction, typically
            /// coordinate axes (e.g. los_direction = {0,0,1} )
            /// We are solving the equation LPT equation
            ///  \f$ \nabla(b\Psi) + \beta \nabla((b\Psi\cdot r)r) = -\delta_{\rm tracer} \f$
            /// and assuming zero-curl (which is not perfectly true) of the second term
            /// so we can use Fourier methods.
            /// Then we subtract the RSD which is given by \f$ \Psi_{\rm rsd} = \beta (b\Psi\cdot r)r) \f$
            /// Here \f$ \beta = f / b \f$ growth-rate over galaxy bias.
            /// The smoothing filter is an (optional) filter we apply to the density field in
            /// k-space. If R < gridspacing then no smoothing will be done
            ///
            // This also works for a survey. Then los_direction is the observer position
            // and we need to have a box big enough so that particles don't wrap around
            // when shifted!
            ///
            /// This method assumes scale-independent growth factor, but that is easily changed
            ///
            /// @tparam N The dimension of the grid
            /// @tparam T The particle class
            ///
            /// @param[out] part MPIParticles. Particles gets updated.
            /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS, PQS)
            /// @param[in] los_direction The fixed line of sight direction, e.g. (0,0,1) for the z-axis
            /// @param[in] Nmesh The size of the grid we use
            /// @param[in] niterations How many iterations to do (not too many and not too few, 3 is good)
            /// @param[in] beta This is beta = f/b the growth-rate over the bias
            /// @param[in] smoothing_options The smoothing filter (gaussian, tophat, sharph) and the smothing scale (in
            /// units of the boxsize)
            /// @param[in] survey_data If survey data we don't wrap around the box if the particles move too far. For
            /// survey data make sure you use padding to prevent any issues..
            ///
            //============================================================================

            template <int N, class T>
            void RSDReconstructionFourierMethod(MPIParticles<T> & part,
                                                std::string density_assignment_method,
                                                std::vector<double> los_direction,
                                                int Nmesh,
                                                int niterations,
                                                double beta,
                                                std::pair<std::string, double> smoothing_options,
                                                bool survey_data) {

                static_assert(FML::PARTICLE::has_get_pos<T>(),
                              "[RSDReconstructionFourierMethod] Particle must have a get_pos method");

                // Use N-linear interpolation
                const std::string interpolation_method = "CIC";

                size_t NumPart = part.get_npart();

                const bool periodic_box = true;

                // Normalize the los_direction to a unit vector
                assert_mpi(los_direction.size() == N,
                           "[RSDReconstructionFourierMethod] Line of sight direction has wrong dimension\n");
                double norm = 0.0;
                for (int idim = 0; idim < N; idim++) {
                    norm += los_direction[idim] * los_direction[idim];
                }
                norm = 1.0 / std::sqrt(norm);
                assert_mpi(norm > 0.0,
                           "[RSDReconstructionFourierMethod] Line of sight vector cannot be the zero vector\n");
                for (int idim = 0; idim < N; idim++) {
                    los_direction[idim] *= norm;
                }

                // Do this iteratively
                for (int i = 0; i < niterations; i++) {
                    if (FML::ThisTask == 0)
                        std::cout << "[RSDReconstructionFourierMethod] Iteration: " << i + 1 << " / " << niterations
                                  << "\n";

                    // This is the density field for the observed galaxies
                    // i.e. with RSD in it
                    auto nleftright =
                        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
                    FFTWGrid<N> density(Nmesh, nleftright.first, nleftright.second);
                    density.add_memory_label("FFTWGrid::RSDReconstructionFourierMethod::density");
                    density.set_grid_status_real(true);
                    FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                                          part.get_npart(),
                                                          part.get_npart_total(),
                                                          density,
                                                          density_assignment_method);

                    density.fftw_r2c();

                    // Perform a smoothing
                    FML::GRID::smoothing_filter_fourier_space(
                        density, smoothing_options.second, smoothing_options.first);

                    // The 1LPT potential
                    FFTWGrid<N> phi_1LPT;
                    compute_1LPT_potential_fourier(density, phi_1LPT);

                    // Free some memory
                    density.free();

                    // The 1LPT displacement field
                    std::array<FFTWGrid<N>, N> Psi;
                    from_LPT_potential_to_displacement_vector<N>(phi_1LPT, Psi);
                    for (int idim = 0; idim < N; idim++) {
                        Psi[idim].communicate_boundaries();
                    }

                    // Free some memory
                    phi_1LPT.free();

                    // Interpolate Psi to particle positions
                    std::array<std::vector<FloatType>, N> Psi_particle_positions;
                    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N,T>(
                        Psi, part.get_particles_ptr(), part.get_npart(), Psi_particle_positions, interpolation_method);
                    for (int idim = 0; idim < N; idim++) {
                        Psi[idim].free();
                    }

                    // Subtract the RSD component (Psi*r)*r / (1+beta) for each particle
                    // Do periodic wrap and communicate particles in case they have left
                    // the current domain
                    std::array<double, N> Psi_max;
                    Psi_max.fill(0.0);
                    auto * p = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++) {
                        auto * pos = FML::PARTICLE::GetPos(p[i]);

                        std::array<FloatType, N> r;
                        if (survey_data) {
                            double norm = 0.0;
                            for (int idim = 0; idim < N; idim++) {
                                r[idim] = pos[idim] - los_direction[idim];
                                norm += r[idim] * r[idim];
                            }
                            norm = 1.0 / std::sqrt(norm);
                            for (int idim = 0; idim < N; idim++) {
                                r[idim] *= norm;
                            }
                        } else {
                            for (int idim = 0; idim < N; idim++) {
                                r[idim] = los_direction[idim];
                            }
                        }

                        std::array<FloatType, N> Psi_rsd;
                        FloatType Psidotr = 0.0;
                        for (int idim = 0; idim < N; idim++) {
                            Psidotr += los_direction[idim] * Psi_particle_positions[idim][i];
                        }
                        for (int idim = 0; idim < N; idim++) {
                            Psi_rsd[idim] = Psidotr * los_direction[idim] / (1.0 + beta);

#ifdef USE_OMP
#pragma omp critical
#endif
                            // Maximum shift
                            if (std::abs(Psi_rsd[idim]) > Psi_max[idim])
                                Psi_max[idim] = std::abs(Psi_rsd[idim]);
                        }

                        // For survey we need to have a box big enough so that we don't wrap around
                        for (int idim = 0; idim < N; idim++) {
                            pos[idim] -= Psi_rsd[idim];
                            if (periodic_box) {
                                if (pos[idim] < 0.0)
                                    pos[idim] += 1.0;
                                if (pos[idim] >= 1.0)
                                    pos[idim] -= 1.0;
                            } else {
                                if (pos[idim] < 0.0 or pos[idim] >= 1.0)
                                    assert_mpi(false,
                                               "[RSDReconstructionFourierMethod] The particles are outside the box "
                                               "and we are set not to periodically wrap");
                            }
                        }
                    }

                    // Particles might have moved out so communicate them
                    part.communicate_particles();

                    // Show maximum shift
                    for (int idim = 0; idim < N; idim++)
                        FML::MaxOverTasks(&Psi_max[idim]);
                    if (FML::ThisTask == 0) {
                        std::cout << "Maximum shift: ";
                        for (int idim = 0; idim < N; idim++)
                            std::cout << Psi_max[idim] << "    ";
                        std::cout << "\n";
                    }
                }
            }
        } // namespace LPT

        // NAMESPACE FML::COSMOLOGY

        //================================================================================
        /// This takes a set of particles and displace them from realspace to redshiftspace
        /// Using a fixed line of sight direction
        /// DeltaX = (v * r)r * velocity_to_displacement
        /// If velocities are peculiar then velocity_to_displacement = 1/(100 * aH/H0 * box) with box in Mpc/h
        /// You can revert back to real-space by calling the method again with velocity_to_displacement being negative
        ///
        /// @tparam T The particle class
        ///
        /// @param[out] part MPIParticle container
        /// @param[in] line_of_sight_direction The fixed line of sight vector, e.g. (0,0,1) for the z-axis
        /// @param[in] velocity_to_displacement Convert from user velocity to RSD shift.
        ///
        //================================================================================
        template <class T>
        void particles_to_redshiftspace(FML::PARTICLE::MPIParticles<T> & part,
                                        std::vector<double> line_of_sight_direction,
                                        double velocity_to_displacement) {

            // Fetch how many dimensjons we are working in
            const int N = FML::PARTICLE::GetNDIM(T());

            // Check that velocities really exists (i.e. get_vel is not just set to return a nullptr)
            static_assert(FML::PARTICLE::has_get_pos<T>(),
                          "[particles_to_redshiftspace] Partices must have positions to use this method");
            static_assert(FML::PARTICLE::has_get_vel<T>(),
                          "[particles_to_redshiftspace] Partices must have velocity to use this method");

            // Periodic box? Yes, this is only meant to be used with simulation boxes
            const bool periodic_box = true;

            // Make sure line_of_sight_direction is a unit vector
            double norm = 0.0;
            for (int idim = 0; idim < N; idim++) {
                norm += line_of_sight_direction[idim] * line_of_sight_direction[idim];
            }
            norm = std::sqrt(norm);
            assert_mpi(norm > 0.0, "[particles_to_redshiftspace] Line of sight vector cannot be the zero vector\n");
            for (int idim = 0; idim < N; idim++) {
                line_of_sight_direction[idim] /= norm;
            }

            double max_disp = 0.0;
            auto NumPart = part.get_npart();
            auto * p = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * pos = FML::PARTICLE::GetPos(p[i]);
                auto * vel = FML::PARTICLE::GetVel(p[i]);
                double vdotr = 0.0;
                for (int idim = 0; idim < N; idim++) {
                    vdotr += vel[idim] * line_of_sight_direction[idim];
                }
                max_disp = std::max(max_disp, std::fabs(vdotr));
                for (int idim = 0; idim < N; idim++) {
                    pos[idim] += vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
                    // Periodic boundary conditions
                    if (periodic_box) {
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                    }
                }
            }

            // Only need to communicate if we have shifted along the x-axis
            if (line_of_sight_direction[0] != 0.0)
                part.communicate_particles();

            // If velocity_to_displacement < 0 we are going the other way around
            FML::MaxOverTasks(&max_disp);
            max_disp *= velocity_to_displacement;
            if (FML::ThisTask == 0)
                std::cout << "[particles_to_redshiftspace] Maximum displacement "
                          << (velocity_to_displacement < 0 ? "(reversing) : " : "            :  ") << max_disp << "\n";
        }

    } // namespace COSMOLOGY
} // namespace FML
#endif
