#ifndef DISPLACEMENTFIELDS_HEADER
#define DISPLACEMENTFIELDS_HEADER
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
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Smoothing/SmoothingFourier.h>

#ifdef USE_GSL
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#endif

namespace FML {
    namespace COSMOLOGY {
        /// This namespace contains things related to Lagrangian Perturbation Theory (displacement fields,
        /// reconstruction, initial conditions etc.)
        namespace LPT {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;
            template <class T>
            using MPIParticles = FML::PARTICLE::MPIParticles<T>;

            template <int N>
            void from_LPT_potential_to_displacement_vector(const FFTWGrid<N> & phi_fourier,
                                                           std::array<FFTWGrid<N>, N> & psi_real,
                                                           double DoverDini = 1.0);

            template <int N>
            void compute_1LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_1LPT_fourier);

            template <int N>
            void compute_2LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_2LPT_fourier);

            // Slightly different syntax here as we have to compute phi_1LPT and phi_2LPT to compute 3LPT so do it all
            template <int N>
            void compute_3LPT_potential_fourier(const FFTWGrid<N> & delta_fourier,
                                                FFTWGrid<N> & phi_1LPT_fourier,
                                                FFTWGrid<N> & phi_2LPT_fourier,
                                                FFTWGrid<N> & phi_3LPT_a_fourier,
                                                FFTWGrid<N> & phi_3LPT_b_fourier,
                                                std::array<FFTWGrid<N>, N> & phi_3LPT_Avec_fourier,
                                                bool ignore_curl_term);

            // Augmented LPT potential (Kitaura and Hess 2013)
            template <int N>
            void compute_ALPT_potential_fourier(const FFTWGrid<N> & delta_fourier,
                                                const FFTWGrid<N> & phi_2LPT_fourier,
                                                FFTWGrid<N> & phi_ALPT_fourier,
                                                double smoothing_scale,
                                                std::string smoothing_method,
                                                double DoverDini_1LPT = 1.0,
                                                double DoverDini_2LPT = 1.0);

            // Spherical collapse approximation potential (Bernardeau 1994)
            template <int N>
            void compute_spherical_collapse_potential(const FFTWGrid<N> & delta_fourier,
                                                      FFTWGrid<N> & phi_sc_fourier,
                                                      double DoverDini = 1.0);

            template <int N>
            void from_LPT_potential_to_displacement_vector_scaledependent(
                const FFTWGrid<N> & phi_fourier,
                std::array<FFTWGrid<N>, N> & psi_real,
                std::function<double(double)> growth_function_ratio);

            // Computes it all for 3LPT as we need lower order results and return Psi and dPsidt
            template <int N>
            void compute_3LPT_displacement_field(const FFTWGrid<N> & delta_fourier,
                                                 std::array<FFTWGrid<N>, N> & Psi,
                                                 std::array<FFTWGrid<N>, N> & dPsidt,
                                                 double dlogDdt,
                                                 bool ignore_curl_term);

            template <int N, class T>
            void assign_displacement_fields_scaledependent(MPIParticles<T> & part,
                                                           FFTWGrid<N> & LPT_potential_fourier,
                                                           std::function<double(double)> DoverDini_of_k,
                                                           std::string LPT_potential_order);

            //=================================================================================
            /// Function is the ratio of the scale-dependent growth-factor at
            /// the time we want to generate particles to the time where phi was generated at
            /// as function of k
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] phi The LPT potential in fourier space
            /// @param[out] psi The displacement vector in real space
            /// @param[in] DoverDini_of_k The function D(k,z)/D(k,zini) as function of k.
            ///
            //=================================================================================
            template <int N>
            void
            from_LPT_potential_to_displacement_vector_scaledependent(const FFTWGrid<N> & phi,
                                                                     std::array<FFTWGrid<N>, N> & psi,
                                                                     std::function<double(double)> DoverDini_of_k) {

                // We require phi to exist and if psi exists it must have the same size as phi
                assert_mpi(phi.get_nmesh() > 0,
                           "[from_LPT_potential_to_displacement_vector_scaledependent] phi grid has to be already "
                           "allocated");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "From LPT potential to displacement field scaledependent\n";
#endif

                auto nleft = phi.get_n_extra_slices_left();
                auto nright = phi.get_n_extra_slices_right();
                auto Nmesh = phi.get_nmesh();
                auto Local_nx = phi.get_local_nx();
                auto Local_x_start = phi.get_local_x_start();

                // Create the output grids if they don't exist already
                for (int idim = 0; idim < N; idim++) {
                    if (psi[idim].get_nmesh() == 0) {
                        psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        psi[idim].add_memory_label(
                            "FFTWGrid::from_LPT_potential_to_displacement_vector_scaledependent::Psi_" +
                            std::to_string(idim));
                        psi[idim].set_grid_status_real(false);
                    }
                }

                // Make a spline of the function (faster) if we have GSL otherwise this is
                // just a copy of the function itself
                auto DoverDini_of_k_spline = phi.make_fourier_spline(DoverDini_of_k, "D(k)/Dini(k)");
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    std::complex<FML::GRID::FloatType> I(0, 1);
                    for (auto && fourier_index : phi.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                        // Psi_vec = D Phi => F[Psi_vec] = ik_vec F[Phi]
                        auto value = phi.get_fourier_from_index(fourier_index) * I * FML::GRID::FloatType(DoverDini_of_k_spline(kmag));

                        for (int idim = 0; idim < N; idim++) {
                            psi[idim].set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(kvec[idim]));
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (int idim = 0; idim < N; idim++)
                        psi[idim].set_fourier_from_index(0, 0.0);

                // Fourier transform Psi
                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transforming Dphi to real space: " << idim + 1 << " / " << N << "\n";
#endif
                    psi[idim].fftw_c2r();
                }
            }

            //=================================================================================
            /// Generate the displaceement field \f$ \Psi = \nabla \phi \f$ from the LPT potential \f$ \phi \f$.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] phi The LPT potential in fourier space
            /// @param[out] psi The displacement vector in real space
            /// @param[in] DoverDini The growth factor at the time you want the displacement field to the growth factor
            /// at which phi is at
            ///
            //=================================================================================
            template <int N>
            void from_LPT_potential_to_displacement_vector(const FFTWGrid<N> & phi,
                                                           std::array<FFTWGrid<N>, N> & psi,
                                                           double DoverDini) {

                // We require phi to exist and if psi exists it must have the same size as phi
                assert_mpi(phi.get_nmesh() > 0,
                           "[from_LPT_potential_to_displacement_vector] Grid has to be already allocated!");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "From LPT potential to displaceent vector\n";
#endif

                auto nleft = phi.get_n_extra_slices_left();
                auto nright = phi.get_n_extra_slices_right();
                auto Nmesh = phi.get_nmesh();
                auto Local_nx = phi.get_local_nx();
                auto Local_x_start = phi.get_local_x_start();

                // Create the output grids if they don't exist already
                for (int idim = 0; idim < N; idim++) {
                    if (psi[idim].get_nmesh() == 0) {
                        psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        psi[idim].add_memory_label("FFTWGrid::from_LPT_potential_to_displacement_vector::Psi_" +
                                                   std::to_string(idim));
                        psi[idim].set_grid_status_real(false);
                    }
                }

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    std::complex<FML::GRID::FloatType> I(0, 1);
                    for (auto && fourier_index : psi[0].get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                        // Psi_vec = D Phi => F[Psi_vec] = ik_vec F[Phi]
                        auto value = phi.get_fourier_from_index(fourier_index) * FML::GRID::FloatType(DoverDini);
                        for (int idim = 0; idim < N; idim++) {
                            psi[idim].set_fourier_from_index(fourier_index, I * value * FML::GRID::FloatType(kvec[idim]));
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (int idim = 0; idim < N; idim++)
                        psi[idim].set_fourier_from_index(0, 0.0);

                // Fourier transform Psi
                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transforming Dphi to real space: " << idim + 1 << " / " << N << "\n";
#endif
                    psi[idim].fftw_c2r();
                }
            }

            //=================================================================================
            /// Generate the 1LPT potential defined as \f$ \Psi^{\rm 1LPT} = \nabla \phi^{\rm 1LPT} \f$ and \f$
            /// \nabla^2 \phi^{\rm 1LPT} = -\delta \f$. Returns it in Fourier space.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta_fourier The density contrast in fourier space
            /// @param[out] phi_1LPT_fourier The LPT potential in fourier space
            ///
            //=================================================================================
            template <int N>
            void compute_1LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_1LPT_fourier) {

                // We require delta to exist and if phi_1LPT is allocated it must have the same size as delta
                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_1LPT_potential_fourier] delta grid has to be already allocated!");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute 1LPT potential\n";
#endif

                auto Nmesh_phi = phi_1LPT_fourier.get_nmesh();

                auto nleft = delta_fourier.get_n_extra_slices_left();
                auto nright = delta_fourier.get_n_extra_slices_right();
                auto Nmesh = delta_fourier.get_nmesh();
                auto Local_nx = delta_fourier.get_local_nx();
                auto Local_x_start = delta_fourier.get_local_x_start();

                // Create 1LPT grid
                if (Nmesh_phi == 0) {
                    phi_1LPT_fourier = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_fourier.add_memory_label("FFTWGrid::compute_1LPT_potential_fourier::phi_1LPT_fourier");
                    phi_1LPT_fourier.set_grid_status_real(false);
                }

                // Divide grid by k^2. Assuming delta was created in fourier-space so no FFTW normalization needed
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        // D^2 Phi_1LPT = -delta => F[Phi_1LPT] = F[delta] / k^2
                        auto value = delta_fourier.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);
                        phi_1LPT_fourier.set_fourier_from_index(fourier_index, value);
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    phi_1LPT_fourier.set_fourier_from_index(0, 0.0);
            }

            //=================================================================================
            /// Generate the 2LPT potential defined as \f$ \Psi^{\rm 2LPT} = \nabla \phi^{\rm 2LPT} \f$ and \f$
            /// \nabla^2 \phi^{\rm 2LPT} = \ldots \f$. Returns the grid in Fourier space.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta The density contrast in fourier space
            /// @param[out] phi_2LPT The LPT potential in fourier space
            ///
            //=================================================================================
            template <int N>
            void compute_2LPT_potential_fourier(const FFTWGrid<N> & delta, FFTWGrid<N> & phi_2LPT) {

                // We require delta to exist and if phi_2LPT is allocated it must have the same size as delta
                assert_mpi(delta.get_nmesh() > 0,
                           "[compute_2LPT_potential_fourier] delta grid has to be already allocated!");

                // This is the -3/7 factor coming from D2 = -3/7 D1^2 for the growing mode in Einstein-deSitter
                constexpr double prefactor_2LPT = -3.0 / 7.0;

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute 2LPT potential (require N(N-1)/2 = 3 for N = 3 temporary grids)\n";
#endif

                auto nleft = delta.get_n_extra_slices_left();
                auto nright = delta.get_n_extra_slices_right();
                auto Nmesh = delta.get_nmesh();
                auto Local_nx = delta.get_local_nx();
                auto Local_x_start = delta.get_local_x_start();

                // Create grids
                FFTWGrid<N> phi_1LPT_ii[N];
                for (int i = 0; i < N; i++) {
                    phi_1LPT_ii[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ii[i].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_1LPT_ii_" +
                                                    std::to_string(i));
                    phi_1LPT_ii[i].set_grid_status_real(false);
                }
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0) {
                    std::cout << "Compute [DiDi phi_1LPT] in fourier space\n";
                }
#endif

                // Compute phi_xx, phi_yy, ...
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_1LPT_ii[0].get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        delta.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        // D^2Phi = -delta => F[DiDj Phi] = F[delta] kikj/k^2
                        auto value = delta.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);

                        for (int idim = 0; idim < N; idim++) {
                            phi_1LPT_ii[idim].set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(kvec[idim] * kvec[idim]));
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (int idim = 0; idim < N; idim++)
                        phi_1LPT_ii[idim].set_fourier_from_index(0, 0.0);

                // Fourier transform
                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transform [DiDi phi_1LPT] to real space: " << idim + 1 << " / " << N
                                  << "\n";
#endif
                    phi_1LPT_ii[idim].fftw_c2r();
                }

                // Crete output grid
                phi_2LPT = FFTWGrid<N>(Nmesh, nleft, nright);
                phi_2LPT.add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_2LPT_fourier");
                phi_2LPT.set_grid_status_real(true);

                // Copy over source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Add 0.5[(D^2 phi_1LPT)^2 - DiDi phi_1LPT^2] to real space grid containing "
                                 "(D^2phi_2LPT) \n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_2LPT.get_real_range(islice, islice + 1)) {
                        auto laplacian = 0.0, sum_squared = 0.0;
                        for (int idim = 0; idim < N; idim++) {
                            auto curpsi = phi_1LPT_ii[idim].get_real_from_index(real_index);
                            laplacian += curpsi;
                            sum_squared += curpsi * curpsi;
                        }
                        auto value = 0.5 * (laplacian * laplacian - sum_squared);
                        phi_2LPT.set_real_from_index(real_index, value);
                    }
                }

                // Free memory
                for (int idim = 0; idim < N; idim++)
                    phi_1LPT_ii[idim].free();

                // Create grids
                const int num_pairs = (N * (N - 1)) / 2;
                FFTWGrid<N> phi_1LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ij[i].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_1LPT_ij_" +
                                                    std::to_string(i));
                    phi_1LPT_ij[i].set_grid_status_real(false);
                }

                // Compute phi_xixj for all pairs of i,j
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute [DiDj phi_1LPT] in fourier space\n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_1LPT_ij[0].get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        delta.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        auto value = delta.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);

                        int pair = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            for (int idim2 = idim1 + 1; idim2 < N; idim2++) {
                                phi_1LPT_ij[pair++].set_fourier_from_index(fourier_index,
                                                                           FML::GRID::FloatType(kvec[idim1] * kvec[idim2]) * value);
                            }
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (auto & g : phi_1LPT_ij)
                        g.set_fourier_from_index(0, 0.0);

                // Fourier transform
                for (int pair = 0; pair < num_pairs; pair++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transform [DiDj phi_1LPT] to real space: " << pair + 1 << " / "
                                  << num_pairs << "\n";
#endif
                    phi_1LPT_ij[pair].fftw_c2r();
                }

                // Copy over source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Add [-DiDjphi_1LPT^2] to real space grid containing (D^2phi_2LPT) \n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_2LPT.get_real_range(islice, islice + 1)) {
                        auto sum_squared = 0.0;
                        for (int pair = 0; pair < num_pairs; pair++) {
                            auto curpsi = phi_1LPT_ij[pair].get_real_from_index(real_index);
                            sum_squared += curpsi * curpsi;
                        }
                        auto value = (phi_2LPT.get_real_from_index(real_index) - sum_squared);

                        phi_2LPT.set_real_from_index(real_index, value);
                    }
                }

                // Free memory
                for (int pair = 0; pair < num_pairs; pair++) {
                    phi_1LPT_ij[pair].free();
                }

                // Fourier transform source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Fourier transform [D^2phi_2LPT] to fourier space\n";
#endif
                phi_2LPT.fftw_r2c();

                // Divide by -k^2 and normalize
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_2LPT in fourier space\n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_2LPT.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_2LPT.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        // Add in the -3/7 factor
                        auto value = phi_2LPT.get_fourier_from_index(fourier_index);
                        value *= -prefactor_2LPT / kmag2;

                        phi_2LPT.set_fourier_from_index(fourier_index, value);
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    phi_2LPT.set_fourier_from_index(0, 0.0);
            }

            //===========================================================================================
            /// In this method we have so far given completely up what we do in the other methods
            /// and try to be as memory efficient as possible. We can reduce the memory footprint of
            /// this method with some work. Right now we allocate ~15 grid at the same time. It is possible to
            /// get that down to ~10. This method is not well tested!
            /// The units of the dlogDdt term is what sets the units of dPsidt
            /// In this methods the displacement field is assumed to be on the EdS/LCDM form
            /// Psi = D Psi1LPT + D^2 Psi2LPT + D^3 Psi3LPT i.e. each term multiplied with powers of D
            /// so we only have a single growth-factor D_nLPT = const * (D_1LPT)^n
            /// with the constants being 1, -3/7, 1/3, -10/21, ...
            ///
            /// @tparam N Dimensions we are working in (only 2 or 3)
            ///
            /// @param[in] delta_fourier A realisation of the density field.
            /// @param[out] Psi The displacement field
            /// @param[out] dPsidt The derivative of the displacement field (units is that of the next factor)
            /// @param[in] dlogDdt The logarithmic derivative of D at the time we want Psi in whatever units you
            /// want it to be.
            /// @param[in] ignore_curl_term
            ///
            //===========================================================================================
            template <int N>
            void compute_3LPT_displacement_field(const FFTWGrid<N> & delta_fourier,
                                                 std::array<FFTWGrid<N>, N> & Psi,
                                                 std::array<FFTWGrid<N>, N> & dPsidt,
                                                 double dlogDdt,
                                                 bool ignore_curl_term) {

                // We require delta to exist
                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_3LPT_displacement_field] delta grid has to be already allocated!");

                auto nleft = delta_fourier.get_n_extra_slices_left();
                auto nright = delta_fourier.get_n_extra_slices_right();
                auto Nmesh = delta_fourier.get_nmesh();
                auto Local_nx = delta_fourier.get_local_nx();
                auto Local_x_start = delta_fourier.get_local_x_start();

                std::array<FFTWGrid<N>, N> phi_3LPT_Avec_fourier;
                FFTWGrid<N> phi_1LPT_fourier;
                FFTWGrid<N> phi_2LPT_fourier;
                FFTWGrid<N> phi_3LPT_a_fourier;
                FFTWGrid<N> phi_3LPT_b_fourier;
                compute_3LPT_potential_fourier(delta_fourier,
                                               phi_1LPT_fourier,
                                               phi_2LPT_fourier,
                                               phi_3LPT_a_fourier,
                                               phi_3LPT_b_fourier,
                                               phi_3LPT_Avec_fourier,
                                               ignore_curl_term);

                // Make the displacment field
                for (int idim = 0; idim < N; idim++) {
                    Psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                    Psi[idim].add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_displacement_field::Psi" +
                                               std::to_string(idim));
                    dPsidt[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                    dPsidt[idim].add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_displacement_field::dPsidt" +
                                                  std::to_string(idim));
                }

                // We compute the displacement field at the initial time
                // so these factors are just 1
                constexpr double DoverDini = 1.0;
                constexpr double DoverDini2 = DoverDini * DoverDini;
                constexpr double DoverDini3 = DoverDini * DoverDini * DoverDini;

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    std::complex<double> I(0, 1);
                    for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                        double fac = -1.0 / kmag2;

                        auto value_1 = phi_1LPT_fourier.get_fourier_from_index(fourier_index);
                        auto value_2 = phi_2LPT_fourier.get_fourier_from_index(fourier_index);
                        auto value_3a = phi_3LPT_a_fourier.get_fourier_from_index(fourier_index);
                        auto value_3b = phi_3LPT_b_fourier.get_fourier_from_index(fourier_index);
                        auto value = -value_1 * DoverDini - 3.0 / 7.0 * value_2 * DoverDini2 +
                                     (value_3a / 3.0 - 10.0 / 21.0 * value_3b) * DoverDini3;
                        auto dvaluedt = -value_1 * DoverDini - 2.0 * 3.0 / 7.0 * value_2 * DoverDini2 +
                                        3.0 * (value_3a / 3.0 - 10.0 / 21.0 * value_3b) * DoverDini3;

                        if constexpr (N == 2) {
                            double Az = 0.0;
                            if (not ignore_curl_term) {
                                Az = phi_3LPT_Avec_fourier[0].get_fourier_from_index(fourier_index);
                            }

                            Psi[0].set_fourier_from_index(
                                fourier_index, (-I * kvec[0] * value + I * DoverDini3 / 7.0 * kvec[1] * Az) * fac);
                            Psi[1].set_fourier_from_index(
                                fourier_index, (-I * kvec[1] * value - I * DoverDini3 / 7.0 * kvec[0] * Az) * fac);

                            fac *= dlogDdt;
                            dPsidt[0].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[0] * dvaluedt + 3.0 * I * DoverDini3 / 7.0 * kvec[1] * Az) * fac);
                            dPsidt[1].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[1] * dvaluedt - 3.0 * I * DoverDini3 / 7.0 * kvec[0] * Az) * fac);

                        } else if (N == 3) {
                            std::array<std::complex<double>, N> A;
                            if (not ignore_curl_term) {
                                A[0] = phi_3LPT_Avec_fourier[0].get_fourier_from_index(fourier_index);
                                A[1] = phi_3LPT_Avec_fourier[1].get_fourier_from_index(fourier_index);
                                A[2] = phi_3LPT_Avec_fourier[2].get_fourier_from_index(fourier_index);
                            } else {
                                A.fill(0.0);
                            }

                            Psi[0].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[0] * value + I * DoverDini3 / 7.0 * (kvec[1] * A[2] - kvec[2] * A[1])) *
                                    fac);
                            Psi[1].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[1] * value + I * DoverDini3 / 7.0 * (kvec[2] * A[0] - kvec[0] * A[2])) *
                                    fac);
                            Psi[2].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[2] * value + I * DoverDini3 / 7.0 * (kvec[0] * A[1] - kvec[1] * A[0])) *
                                    fac);

                            fac *= dlogDdt;
                            dPsidt[0].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[0] * dvaluedt +
                                 3.0 * I * DoverDini3 / 7.0 * (kvec[1] * A[2] - kvec[2] * A[1])) *
                                    fac);
                            dPsidt[1].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[1] * dvaluedt +
                                 3.0 * I * DoverDini3 / 7.0 * (kvec[2] * A[0] - kvec[0] * A[2])) *
                                    fac);
                            dPsidt[2].set_fourier_from_index(
                                fourier_index,
                                (-I * kvec[2] * dvaluedt +
                                 3.0 * I * DoverDini3 / 7.0 * (kvec[0] * A[1] - kvec[1] * A[0])) *
                                    fac);
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0) {
                    for (auto & g : Psi)
                        g.set_fourier_from_index(0, 0.0);
                    for (auto & g : dPsidt)
                        g.set_fourier_from_index(0, 0.0);
                }

                // Fourier transform to real space and we are done
                for (int idim = 0; idim < N; idim++) {
                    Psi[idim].fftw_c2r();
                    dPsidt[idim].fftw_c2r();
                }
            }

            //===========================================================================================
            /// In this method we have so far given completely up what we do in the other methods
            /// and try to be as memory efficient as possible. We can reduce the memory footprint of
            /// this method with some work. Right now we allocate ~15 grid at the same time. It is possible to
            /// get that down to ~10. This method is not well tested!
            /// The potentials we output are normalized such that we can get the displacement field as
            /// Psi = D phi_1LPT + D phi_2LPT + D phi_3LPT_a + D phi_3PT_b + D x A_3LPT
            ///
            /// @tparam N Dimensions we are working in (only 2 or 3)
            ///
            /// @param[in] delta_fourier A realisation of the density field.
            /// @param[out] phi_1LPT_fourier The 1LPT displacement potential
            /// @param[out] phi_2LPT_fourier The 2LPT displacement potential
            /// @param[out] phi_3LPT_a_fourier The A 3LPT displacement potential
            /// @param[out] phi_3LPT_b_fourier The B 3LPT displacement potential
            /// @param[out] phi_3LPT_Avec_fourier The 3LPT displacement vector potential
            /// @param[in] ignore_curl_term Don't compute the vector potential
            ///
            //===========================================================================================
            template <int N>
            void compute_3LPT_potential_fourier(const FFTWGrid<N> & delta_fourier,
                                                FFTWGrid<N> & phi_1LPT_fourier,
                                                FFTWGrid<N> & phi_2LPT_fourier,
                                                FFTWGrid<N> & phi_3LPT_a_fourier,
                                                FFTWGrid<N> & phi_3LPT_b_fourier,
                                                std::array<FFTWGrid<N>, N> & phi_3LPT_Avec_fourier,
                                                bool ignore_curl_term) {

                // Only works for N = 2 and N = 3
                static_assert(N == 2 or N == 3);

                // We require delta to exist
                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_3LPT_displacement_field] delta grid has to be already allocated!");

                // Factor to scale displacement potentials such that Psi = Dphi_1LPT + Dphi_2LPT + ... + D x Avec_3LPT
                constexpr FML::GRID::FloatType prefactor_1LPT = -1.0;
                constexpr FML::GRID::FloatType prefactor_2LPT = -3.0 / 7.0;
                constexpr FML::GRID::FloatType prefactor_3LPT_a = 1.0 / 3.0;
                constexpr FML::GRID::FloatType prefactor_3LPT_b = -10.0 / 21.0;
                constexpr FML::GRID::FloatType prefactor_3LPT_Avec = 1.0 / 7.0;

                auto nleft = delta_fourier.get_n_extra_slices_left();
                auto nright = delta_fourier.get_n_extra_slices_right();
                auto Nmesh = delta_fourier.get_nmesh();
                auto Local_nx = delta_fourier.get_local_nx();
                auto Local_x_start = delta_fourier.get_local_x_start();

                // Store -k^2phi_1LPT
                phi_1LPT_fourier = delta_fourier;
                phi_1LPT_fourier.add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_1LPT_fourier");

                // Compute all terms phi_1LPT_ij. These are absolutely needed
                const int num_pairs = (N * (N + 1)) / 2;
                FFTWGrid<N> phi_1LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ij[i].add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_1LPT_ij_" +
                                                    std::to_string(i));
                }

                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_1LPT_ij for all i,j...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        // phi_1LPT_fourier is delta so transform to LPT potential D^2 phi_1LPT = -delta
                        auto value = -phi_1LPT_fourier.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);

                        int pair = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            for (int idim2 = idim1; idim2 < N; idim2++) {
                                phi_1LPT_ij[pair++].set_fourier_from_index(fourier_index,
                                                                           -value * FML::GRID::FloatType(kvec[idim1] * kvec[idim2]));
                            }
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (auto & g : phi_1LPT_ij)
                        g.set_fourier_from_index(0, 0.0);

                // Fourier transform it all to real-space
                for (int i = 0; i < num_pairs; i++)
                    phi_1LPT_ij[i].fftw_c2r();

                // Compute 2LPT
                phi_2LPT_fourier = FFTWGrid<N>(Nmesh, nleft, nright);
                phi_2LPT_fourier.add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_2LPT_fourier");

                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_2LPT...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_2LPT_fourier.get_real_range(islice, islice + 1)) {
                        // Compute laplacian and sum of squares to get Sum_i,j Phi_iiPhi_jj Phi_ij^2
                        double laplacian = 0.0;
                        double sum_squared = 0.0;
                        int pair = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            auto phi_ij = phi_1LPT_ij[pair++].get_real_from_index(real_index);
                            laplacian += phi_ij;
                            sum_squared += phi_ij * phi_ij;
                            for (int idim2 = idim1 + 1; idim2 < N; idim2++) {
                                phi_ij = phi_1LPT_ij[pair++].get_real_from_index(real_index);
                                sum_squared += 2.0 * phi_ij * phi_ij;
                            }
                        }
                        phi_2LPT_fourier.set_real_from_index(real_index, 0.5 * (laplacian * laplacian - sum_squared));
                    }
                }

                // Back to fourier space: We now have -k^2 phi_2LPT in this grid
                phi_2LPT_fourier.fftw_r2c();

                // Time to compute all the phi_2LPT_ij terms
                FFTWGrid<N> phi_2LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_2LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_2LPT_ij[i].add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_2LPT_ij" +
                                                    std::to_string(i));
                }

                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_2LPT_ij for all i,j...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_2LPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_2LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                        auto value = -phi_2LPT_fourier.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);

                        int pair = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            for (int idim2 = idim1; idim2 < N; idim2++) {
                                phi_2LPT_ij[pair++].set_fourier_from_index(fourier_index,
                                                                           -value * FML::GRID::FloatType(kvec[idim1] * kvec[idim2]));
                            }
                        }
                    }
                }

                // Deal with DC mode
                if (Local_x_start == 0)
                    for (auto & g : phi_2LPT_ij)
                        g.set_fourier_from_index(0, 0.0);

                // Compute phi_3LPT_a
                phi_3LPT_a_fourier = FFTWGrid<N>(Nmesh, nleft, nright);
                phi_3LPT_a_fourier.add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_3LPT_a_fourier");
                // Compute phi_3LPT_b
                phi_3LPT_b_fourier = FFTWGrid<N>(Nmesh, nleft, nright);
                phi_3LPT_b_fourier.add_memory_label("FFTWGrid::compute_3LPT_potential_fourier::phi_3LPT_b_fourier");
                // And then finally the A-terms (for N=2 we only have 1 component)
                if (not ignore_curl_term)
                    for (int idim = 0; idim < N; idim++) {
                        phi_3LPT_Avec_fourier[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        phi_3LPT_Avec_fourier[idim].add_memory_label(
                            "FFTWGrid::compute_3LPT_potential_fourier::phi_3LPT_Avec_fourier" + std::to_string(idim));
                        if constexpr (N == 2)
                            break;
                    }

                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_3LPT...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_2LPT_fourier.get_real_range(islice, islice + 1)) {

                        if constexpr (N == 2) {
                            auto psi1_xx = phi_1LPT_ij[0].get_real_from_index(real_index);
                            auto psi1_xy = phi_1LPT_ij[1].get_real_from_index(real_index);
                            auto psi1_yy = phi_1LPT_ij[2].get_real_from_index(real_index);
                            auto psi2_xx = phi_2LPT_ij[0].get_real_from_index(real_index);
                            auto psi2_xy = phi_2LPT_ij[1].get_real_from_index(real_index);
                            auto psi2_yy = phi_2LPT_ij[2].get_real_from_index(real_index);

                            auto value_a = psi1_xx * psi1_yy - psi1_xy * psi1_xy;
                            auto value_b = psi1_xx * psi2_yy - psi1_xy * psi2_xy;

                            phi_3LPT_a_fourier.set_real_from_index(real_index, value_a);
                            phi_3LPT_b_fourier.set_real_from_index(real_index, value_b);
                            if (not ignore_curl_term) {
                                // Curl is a scalar in 2D (only the 'z' component is nonzero)
                                auto Az = psi2_xy * (psi1_yy - psi1_xx) - psi1_xy * (psi2_yy - psi2_xx);
                                phi_3LPT_Avec_fourier[0].set_real_from_index(real_index, Az);
                            }
                        }
                        if constexpr (N == 3) {
                            auto psi1_xx = phi_1LPT_ij[0].get_real_from_index(real_index);
                            auto psi1_xy = phi_1LPT_ij[1].get_real_from_index(real_index);
                            auto psi1_zx = phi_1LPT_ij[2].get_real_from_index(real_index);
                            auto psi1_yy = phi_1LPT_ij[3].get_real_from_index(real_index);
                            auto psi1_yz = phi_1LPT_ij[4].get_real_from_index(real_index);
                            auto psi1_zz = phi_1LPT_ij[5].get_real_from_index(real_index);

                            auto psi2_xx = phi_2LPT_ij[0].get_real_from_index(real_index);
                            auto psi2_xy = phi_2LPT_ij[1].get_real_from_index(real_index);
                            auto psi2_zx = phi_2LPT_ij[2].get_real_from_index(real_index);
                            auto psi2_yy = phi_2LPT_ij[3].get_real_from_index(real_index);
                            auto psi2_yz = phi_2LPT_ij[4].get_real_from_index(real_index);
                            auto psi2_zz = phi_2LPT_ij[5].get_real_from_index(real_index);

                            auto value_a = psi1_xx * psi1_yy * psi1_zz;
                            value_a += 2.0 * psi1_xy * psi1_yz * psi1_zx;
                            value_a += -psi1_xx * psi1_yz * psi1_yz;
                            value_a += -psi1_yy * psi1_zx * psi1_zx;
                            value_a += -psi1_zz * psi1_xy * psi1_xy;

                            auto value_b = 0.5 * psi1_xx * (psi2_yy + psi2_zz);
                            value_b += 0.5 * psi1_yy * (psi2_zz + psi2_xx);
                            value_b += 0.5 * psi1_zz * (psi2_xx + psi2_yy);
                            value_b += -psi1_xy * psi2_xy - psi1_yz * psi2_yz - psi1_zx * psi2_zx;

                            phi_3LPT_a_fourier.set_real_from_index(real_index, value_a);
                            phi_3LPT_b_fourier.set_real_from_index(real_index, value_b);
                            if (not ignore_curl_term) {
                                auto value_Avec_x = psi1_zx * psi2_xy - psi2_zx * psi1_xy;
                                value_Avec_x += psi1_yz * (psi2_yy - psi2_zz) - psi2_yz * (psi1_yy - psi1_zz);
                                auto value_Avec_y = psi1_xy * psi2_yz - psi2_xy * psi1_yz;
                                value_Avec_y += psi1_zx * (psi2_zz - psi2_xx) - psi2_zx * (psi1_zz - psi1_xx);
                                auto value_Avec_z = psi1_yz * psi2_zx - psi2_yz * psi1_zx;
                                value_Avec_z += psi1_xy * (psi2_xx - psi2_yy) - psi2_xy * (psi1_xx - psi1_yy);
                                phi_3LPT_Avec_fourier[0].set_real_from_index(real_index, value_Avec_x);
                                phi_3LPT_Avec_fourier[1].set_real_from_index(real_index, value_Avec_y);
                                phi_3LPT_Avec_fourier[2].set_real_from_index(real_index, value_Avec_z);
                            }
                        }
                    }
                }

                // Free up memory
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i].free();
                    phi_2LPT_ij[i].free();
                }

                // Fourier transform and voila we have -k^2phi_3LPT_a stored in phi_3LPT_a
                phi_3LPT_a_fourier.fftw_r2c();

                // Fourier transform and voila we have -k^2phi_3LPT_b stored in phi_3LPT_b
                phi_3LPT_b_fourier.fftw_r2c();

                // Fourier transform and voila we have -k^2phi_3LPT_Avec stored in phi_3LPT_Avec
                if (not ignore_curl_term) {
                    for (int idim = 0; idim < N; idim++)
                        phi_3LPT_Avec_fourier[idim].fftw_r2c();
                }

                // Divide by -1/k^2 and multiply by factor to make Psi = Dphi^1LPT + Dphi^2LPT + Dphi^3LPT + D x
                // Avec^3LPT
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)

                        // Get wavevector and magnitude
                        phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                        const FML::GRID::FloatType fac = -1.0 / kmag2;

                        // Fetch and rescale
                        auto value_1LPT = phi_1LPT_fourier.get_fourier_from_index(fourier_index);
                        auto value_2LPT = phi_2LPT_fourier.get_fourier_from_index(fourier_index);
                        auto value_3LPT_a = phi_3LPT_a_fourier.get_fourier_from_index(fourier_index);
                        auto value_3LPT_b = phi_3LPT_b_fourier.get_fourier_from_index(fourier_index);
                        phi_1LPT_fourier.set_fourier_from_index(fourier_index, prefactor_1LPT * value_1LPT * fac);
                        phi_2LPT_fourier.set_fourier_from_index(fourier_index, prefactor_2LPT * value_2LPT * fac);
                        phi_3LPT_a_fourier.set_fourier_from_index(fourier_index, prefactor_3LPT_a * value_3LPT_a * fac);
                        phi_3LPT_b_fourier.set_fourier_from_index(fourier_index, prefactor_3LPT_b * value_3LPT_b * fac);

                        // The vector potential A
                        if (not ignore_curl_term) {
                            if constexpr (N == 2) {
                                // For N=2 the curl is a scalar and we only store the "z" component
                                auto value_Az = phi_3LPT_Avec_fourier[0].get_fourier_from_index(fourier_index);
                                phi_3LPT_Avec_fourier[0].set_fourier_from_index(fourier_index,
                                                                                prefactor_3LPT_Avec * value_Az * fac);
                            }
                            if constexpr (N == 3) {
                                auto value_Ax = phi_3LPT_Avec_fourier[0].get_fourier_from_index(fourier_index);
                                auto value_Ay = phi_3LPT_Avec_fourier[1].get_fourier_from_index(fourier_index);
                                auto value_Az = phi_3LPT_Avec_fourier[2].get_fourier_from_index(fourier_index);
                                phi_3LPT_Avec_fourier[0].set_fourier_from_index(fourier_index,
                                                                                prefactor_3LPT_Avec * value_Ax * fac);
                                phi_3LPT_Avec_fourier[1].set_fourier_from_index(fourier_index,
                                                                                prefactor_3LPT_Avec * value_Ay * fac);
                                phi_3LPT_Avec_fourier[2].set_fourier_from_index(fourier_index,
                                                                                prefactor_3LPT_Avec * value_Az * fac);
                            }
                        }
                    }
                }
            }

            //=================================================================================
            /// Take in an initial density field (generated at a redshift zini) and the corresponding ratio of growth
            /// factors D(z)/D(zini) produces the approximate spherical collapse potential defined via D^2 phi_SC = 3(
            /// (1
            /// - 2/3 * phi_1LPT(x,z))^0.5 - 1) at the redshift z
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta_fourier The density contrast in fourier space
            /// @param[out] phi_sc_fourier The spherical collapse approximation potential in fourier space
            /// @param[in] DoverDini_1LPT The 1LPT growth-factor at the time we want phi_ALPT_fourier divide by the
            /// growth factor at the redshift delta_fourier is at
            ///
            //=================================================================================
            template <int N>
            void compute_spherical_collapse_potential(const FFTWGrid<N> & delta_fourier,
                                                      FFTWGrid<N> & phi_sc_fourier,
                                                      double DoverDini_1LPT) {

                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_spherical_collapse_potential] delta_fourier grid has to be already allocated");

                auto Nmesh = delta_fourier.get_nmesh();
                auto Local_nx = delta_fourier.get_local_nx();
                auto Local_x_start = delta_fourier.get_local_x_start();

                phi_sc_fourier = delta_fourier;
                phi_sc_fourier.add_memory_label("FFTWGrid::compute_spherical_collapse_potential::phi_sc_fourier");
                phi_sc_fourier.set_grid_status_real(false);
                phi_sc_fourier.fftw_c2r();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_sc_fourier.get_real_range(islice, islice + 1)) {
                        auto delta_ini = phi_sc_fourier.get_real_from_index(real_index);
                        double value = 1.0 - 2.0 / 3.0 * delta_ini * DoverDini_1LPT;
                        value = std::max(value, 0.0);
                        phi_sc_fourier.set_real_from_index(real_index, 3.0 * (std::sqrt(value) - 1.0));
                    }
                }
                phi_sc_fourier.fftw_r2c();

                // Set the DC mode
                if (Local_x_start == 0)
                    phi_sc_fourier.set_real_from_index(0, 0.0);
            }

            //=================================================================================
            /// Generate the APT potential defined as \f$ \Psi = \Psi^L + \Psi^S \f$ where the
            /// short range is just a smoothed 1LPT+2LPT and the short range is the spherical collapse
            /// approximation.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta_fourier The density contrast in fourier space
            /// @param[in] phi_2LPT_fourier The 2LPT potential in fourier space
            /// @param[out] phi_ALPT_fourier The ALPT potential in fourier space
            /// @param[in] smoothing_scale The smoothing scale in the unit of the boxsize
            /// @param[in] smoothing_method The smoothing filter (gaussian, tophat, sharpk)
            /// @param[in] DoverDini_1LPT The 1LPT growth-factor at the time we want phi_ALPT_fourier divide by the
            /// growth factor at the redshift delta_fourier is at
            /// @param[in] DoverDini_2LPT The 2LPT growth-factor at the time we want phi_ALPT_fourier divide by the
            /// growth factor at the redshift delta_fourier is at
            ///
            //=================================================================================
            template <int N>
            void compute_ALPT_potential_fourier(const FFTWGrid<N> & delta_fourier,
                                                const FFTWGrid<N> & phi_2LPT_fourier,
                                                FFTWGrid<N> & phi_ALPT_fourier,
                                                double smoothing_scale,
                                                std::string smoothing_method,
                                                double DoverDini_1LPT,
                                                double DoverDini_2LPT) {

                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_ALPT_potential_fourier] delta_fourier grid has to be already allocated");
                assert_mpi(phi_2LPT_fourier.get_nmesh() > 0,
                           "[compute_ALPT_potential_fourier] phi_2LPT_fourier grid has to be already allocated");
                assert_mpi(phi_2LPT_fourier.get_nmesh() == delta_fourier.get_nmesh(),
                           "[compute_ALPT_potential_fourier] phi_2LPT_fourier grid  and delta_fourier must have the "
                           "same size");

                auto Nmesh = delta_fourier.get_nmesh();
                auto Local_nx = delta_fourier.get_local_nx();
                auto Local_x_start = delta_fourier.get_local_x_start();

                // Compute the short range spherical collapse approximation
                // D*Psi_SC = 3( sqrt(1 - 2/3 * D/Dini delta_ini) - 1 )
                FFTWGrid<N> phi_sc_fourier;
                compute_spherical_collapse_potential(delta_fourier, phi_sc_fourier, DoverDini_1LPT);

                // Copy over the long range 2LPT approximation
                phi_ALPT_fourier = phi_2LPT_fourier;
                phi_ALPT_fourier.add_memory_label("FFTWGrid::compute_ALPT_potential_fourier::phi_ALPT_fourier");
                phi_ALPT_fourier.set_grid_status_real(false);

                // Compute (phi_1LPT + phi_2LPT - phi_SC)
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag2;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_ALPT_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode (k=0)
                        auto short_range = phi_sc_fourier.get_fourier_from_index(fourier_index);
                        auto p1LPT = delta_fourier.get_fourier_from_index(fourier_index) / kmag2;
                        auto p2LPT = phi_ALPT_fourier.get_fourier_from_index(fourier_index);
                        auto long_range = p1LPT * DoverDini_1LPT + p2LPT * DoverDini_2LPT;
                        phi_ALPT_fourier.set_fourier_from_index(fourier_index, long_range - short_range);
                    }
                }

                // Smooth (phi_1LPT + phi_2LPT - phi_SC)
                FML::GRID::smoothing_filter_fourier_space(phi_ALPT_fourier, smoothing_scale, smoothing_method);

                // Set phi_ALPT = phi_SC + Smooth(phi_1LPT + phi_2LPT - phi_SC)
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && fourier_index : phi_ALPT_fourier.get_fourier_range(islice, islice + 1)) {
                        auto short_range = phi_sc_fourier.get_fourier_from_index(fourier_index);
                        auto smoothed = phi_ALPT_fourier.get_fourier_from_index(fourier_index);
                        phi_ALPT_fourier.set_fourier_from_index(fourier_index, short_range + smoothed);
                    }
                }
            }

#ifdef USE_GSL

            //=================================================================================
            /// Compute 1,2,3LPT growth-factors in simple modified gravity models that have a GeffG(a) and spline them
            /// The growth factors are normalized such that D1LPT = 1 at zini
            /// We assume initial conditions as in EdS: D1LPT=1, D2LPT = -3/7, D3LPTa = -1/3 and D3LPTb = 5/21
            /// NB: There should be a modified 2LPT+ kernel factor in general which we don't include here (model
            /// dependent)
            ///
            /// @param[in] OmegaM Matter density parameter at z=0
            /// @param[in] zini The redshift we want D1LPT = 1
            /// @param[in] HoverH0_of_a The hubble function H(a)/H0
            /// @param[in] GeffOverG_of_a Newtons constant Geff/G as function of a
            /// @param[out] D_1LPT_of_loga Spline of the 1LPT growth factor
            /// @param[out] D_2LPT_of_loga Spline of the 2LPT growth factor
            /// @param[out] D_3LPTa_of_loga Spline of the 3LPTb growth factor
            /// @param[out] D_3LPTb_of_loga Spline of the 3LPTa growth factor
            ///
            //=================================================================================
            void compute_LPT_growth_factors_GeffGLCDM(double OmegaM,
                                                      double zini,
                                                      std::function<double(double)> HoverH0_of_a,
                                                      std::function<double(double)> GeffOverG_of_a,
                                                      FML::INTERPOLATION::SPLINE::Spline & D_1LPT_of_loga,
                                                      FML::INTERPOLATION::SPLINE::Spline & D_2LPT_of_loga,
                                                      FML::INTERPOLATION::SPLINE::Spline & D_3LPTa_of_loga,
                                                      FML::INTERPOLATION::SPLINE::Spline & D_3LPTb_of_loga) {

                using DVector = std::vector<double>;

                // Start at z = 1000 assuming no radiation, but this is fine because we want the solution to follow
                // the matter attractor
                const int npts = 2000;
                const double aini = 1.0 / 1000.0;
                const double aend = 1.0;

                FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                    const double a = std::exp(x);
                    const double H = HoverH0_of_a(a);
                    const double dlogHdx = 1.0 / (2.0 * H * H) * (-3.0 * OmegaM / (a * a * a));
                    const double factor = 1.5 * OmegaM * GeffOverG_of_a(a) / (H * H * a * a * a);
                    const double D1 = y[0];
                    const double dD1dx = y[1];
                    const double D2 = y[2];
                    const double dD2dx = y[3];
                    const double D3a = y[4];
                    const double dD3adx = y[5];
                    const double D3b = y[6];
                    const double dD3bdx = y[7];
                    dydx[0] = dD1dx;
                    dydx[1] = factor * D1 - (2.0 + dlogHdx) * dD1dx;
                    dydx[2] = dD2dx;
                    dydx[3] = factor * (D2 - D1 * D1) - (2.0 + dlogHdx) * dD2dx;
                    dydx[4] = dD3adx;
                    dydx[5] = factor * (D3a - 2.0 * D1 * D1 * D1) - (2.0 + dlogHdx) * dD3adx;
                    dydx[6] = dD3bdx;
                    dydx[7] = factor * (D3b + D1 * D1 * D1 - D2 * D1) - (2.0 + dlogHdx) * dD3bdx;
                    return GSL_SUCCESS;
                };

                const double D1_ini = 1.0;
                const double dD1dx_ini = 1.0 * 1.0;
                const double D2_ini = -3.0 / 7.0;
                const double dD2dx_ini = -3.0 / 7.0 * 2.0;
                const double D3a_ini = -1.0 / 3.0;
                const double dD3adx_ini = -1.0 / 3.0 * 3.0;
                const double D3b_ini = 5.0 / 21.0;
                const double dD3bdx_ini = 5.0 / 21.0 * 3.0;

                // The initial conditions
                // D1 = a/aini, D2 = -3/7 D1^2, D3a = -1/3 D1^3 and D3b = 5/21 D1^3 for growing mode in EdS
                DVector yini{D1_ini, dD1dx_ini, D2_ini, dD2dx_ini, D3a_ini, dD3adx_ini, D3b_ini, dD3bdx_ini};
                DVector xarr(npts);
                for (int i = 0; i < npts; i++)
                    xarr[i] = std::log(aini) + std::log(aend / aini) * i / double(npts);

                // Solve the ODE
                FML::SOLVERS::ODESOLVER::ODESolver ode;
                ode.solve(deriv, xarr, yini);
                auto D1 = ode.get_data_by_component(0);
                auto D2 = ode.get_data_by_component(2);
                auto D3a = ode.get_data_by_component(4);
                auto D3b = ode.get_data_by_component(6);

                FML::INTERPOLATION::SPLINE::Spline tmp;
                tmp.create(xarr, D1, "D1(loga) Spline");
                const double D = tmp(std::log(1.0 / (1.0 + zini)));

                // Normalize such that D1LPT = 1 at zini
                for (int i = 0; i < npts; i++) {
                    D1[i] /= D;
                    D2[i] /= D * D;
                    D3a[i] /= D * D * D;
                    D3b[i] /= D * D * D;
                }

                // Print some values
                if (FML::ThisTask == 0) {
                    std::cout << "[compute_LPT_growth_factors_LCDM]  Scalefactor   Growth factors (D1LPT / a,   "
                                 "D2LPT / D1LPT^2,   D3LPTa / D1LPT^3,   D3LPTb / "
                                 "D1LPT^3)\n";
                    for (int i = 0; i < npts; i += npts / 15)
                        std::cout << std::exp(xarr[i]) << " " << D1[i] / (1.0 + zini) / std::exp(xarr[i]) << " "
                                  << D2[i] / (D1[i] * D1[i]) << " " << D3a[i] / (D1[i] * D1[i] * D1[i]) << " "
                                  << D3b[i] / (D1[i] * D1[i] * D1[i]) << " "
                                  << "\n";
                }

                // Spline it up
                D_1LPT_of_loga.create(xarr, D1, "D1(loga) Spline");
                D_2LPT_of_loga.create(xarr, D2, "D2(loga) Spline");
                D_3LPTa_of_loga.create(xarr, D3a, "D3a(loga) Spline");
                D_3LPTb_of_loga.create(xarr, D3b, "D3b(loga) Spline");
            }

            //=================================================================================
            /// Compute 1,2,3LPT growth-factors in LCDM and spline them
            /// The growth factors are normalized such that D1LPT = 1 at zini
            /// We assume initial conditions as in EdS: D1LPT=1, D2LPT = -3/7, D3LPTa = -1/3 and D3LPTb = 5/21
            ///
            /// @param[in] OmegaM Matter density parameter at z=0
            /// @param[in] zini The redshift we want D1LPT = 1
            /// @param[in] HoverH0_of_a The hubble function H(a)/H0
            /// @param[out] D_1LPT_of_loga Spline of the 1LPT growth factor
            /// @param[out] D_2LPT_of_loga Spline of the 2LPT growth factor
            /// @param[out] D_3LPTa_of_loga Spline of the 3LPTb growth factor
            /// @param[out] D_3LPTb_of_loga Spline of the 3LPTa growth factor
            ///
            //=================================================================================
            void compute_LPT_growth_factors_LCDM(double OmegaM,
                                                 double zini,
                                                 std::function<double(double)> HoverH0_of_a,
                                                 FML::INTERPOLATION::SPLINE::Spline & D_1LPT_of_loga,
                                                 FML::INTERPOLATION::SPLINE::Spline & D_2LPT_of_loga,
                                                 FML::INTERPOLATION::SPLINE::Spline & D_3LPTa_of_loga,
                                                 FML::INTERPOLATION::SPLINE::Spline & D_3LPTb_of_loga) {

                auto GeffOverG_of_a = []([[maybe_unused]] double a) { return 1.0; };
                compute_LPT_growth_factors_GeffGLCDM(OmegaM,
                                                     zini,
                                                     HoverH0_of_a,
                                                     GeffOverG_of_a,
                                                     D_1LPT_of_loga,
                                                     D_2LPT_of_loga,
                                                     D_3LPTa_of_loga,
                                                     D_3LPTb_of_loga);
            }

#endif

            //===================================================================================
            /// This method assigns the displacement fields to particles for the case where we have scaledependent
            /// growth. This is involved: we take phi_iLPT(zini,k) and use growth factors to go to phi_iLPT(z,k) and FFT
            /// to get phi_iLPT(z,q). Then we transport the particles back to their original CPU given by the particles
            /// Lagrangian position q and assign the displacement field before we move them back to their eulerian
            /// positions. This can be done more efficiently (we don't need to do this much communiation) but this will
            /// come at the expense of a lot more code so fuck it.
            /// In practice to be efficient we should ideally do *all* orders at the same time otherwise we need to
            /// swap and do communication many times. This is how its done in our COLA code. So this method mainly
            /// shows how to do it.
            ///
            /// @tparam N The dimension we are working in
            /// @tparam T The particle class. Must have a get_D_* with * being one or more of 1LPT, 2LPT, 3LPTa, 3LPTb
            /// to use this method.
            ///
            /// @param[out] part The particle container. We assign the displacement fields to the particles we get in.
            /// @param[in] LPT_potential_fourier The LPT potential phi(zini,k) at the initial redshift.
            /// @param[in] DoverDini_of_k Ratio of the growth factors at the current time to that of the time of
            /// LPT_potential_fourier.
            /// @param[in] LPT_potential_order 1LPT, 2LPT, 3LPTa or 3LPTb. Tells us what field in the particles we
            /// should store the data in
            ///
            //===================================================================================
            template <int N, class T>
            void assign_displacement_fields_scaledependent(MPIParticles<T> & part,
                                                           FFTWGrid<N> & LPT_potential_fourier,
                                                           std::function<double(double)> DoverDini_of_k,
                                                           std::string LPT_potential_order) {
                enum Order { _1LPT, _2LPT, _3LPTA, _3LPTB };
                const std::string interpolation_method = "CIC";
                int LPT_order{};

                // Sanity checks
                assert_mpi(
                    FML::PARTICLE::has_get_q<T>(),
                    "[assign_displacement_fields_scaledependent] Particle must have Lagrangian position to use this");
                if (LPT_potential_order == "1LPT") {
                    LPT_order = _1LPT;
                    assert_mpi(FML::PARTICLE::has_get_D_1LPT<T>(),
                               "[assign_displacement_fields_scaledependent] Particle must have D_1LPT to use this");
                } else if (LPT_potential_order == "2LPT") {
                    LPT_order = _2LPT;
                    assert_mpi(FML::PARTICLE::has_get_D_2LPT<T>(),
                               "[assign_displacement_fields_scaledependent] Particle must have D_2LPT to use this");
                } else if (LPT_potential_order == "3LPTa") {
                    LPT_order = _3LPTA;
                    assert_mpi(FML::PARTICLE::has_get_D_3LPTa<T>(),
                               "[assign_displacement_fields_scaledependent] Particle must have D_3LPTa to use this");
                } else if (LPT_potential_order == "3LPTb") {
                    LPT_order = _3LPTB;
                    assert_mpi(FML::PARTICLE::has_get_D_3LPTb<T>(),
                               "[assign_displacement_fields_scaledependent] Particle must have D_3LPTb to use this");
                } else {
                    assert_mpi(false,
                               "[assign_displacement_fields_scaledependent] Unknown LPT_potential, options "
                               "1LPT,2LPT,3LPTa,3LPTb");
                }

                // Swap positions with Lagrangian position
                // Bring particles to the task they started off from
                FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
                part.communicate_particles();

                // Use initial LPT potential + growth factor to compute displacement field at present time
                std::array<FFTWGrid<N>, N> psi_LPT_vector;
                FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector_scaledependent(
                    LPT_potential_fourier, psi_LPT_vector, DoverDini_of_k);
                for (int idim = 0; idim < N; idim++) {
                    psi_LPT_vector[idim].communicate_boundaries();
                }
                // Interpolate to particle positions after which we have Psi(q,t) in displacements
                std::array<std::vector<FML::GRID::FloatType>, N> displacements;
                FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions(
                    psi_LPT_vector, part.get_particles_ptr(), part.get_npart(), displacements, interpolation_method);
                for (int idim = 0; idim < N; idim++) {
                    psi_LPT_vector[idim].free();
                }

                // Assign displacment field to particles
                auto np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < np; ind++) {
                    if (LPT_order == _1LPT) {
                        auto * D = FML::PARTICLE::GetD_1LPT(part[ind]);
                        for (int idim = 0; idim < N; idim++)
                            D[idim] = displacements[idim][ind];
                    } else if (LPT_order == _2LPT) {
                        auto * D = FML::PARTICLE::GetD_2LPT(part[ind]);
                        for (int idim = 0; idim < N; idim++)
                            D[idim] = displacements[idim][ind];
                    } else if (LPT_order == _3LPTA) {
                        auto * D = FML::PARTICLE::GetD_3LPTa(part[ind]);
                        for (int idim = 0; idim < N; idim++)
                            D[idim] = displacements[idim][ind];
                    } else if (LPT_order == _3LPTB) {
                        auto * D = FML::PARTICLE::GetD_3LPTb(part[ind]);
                        for (int idim = 0; idim < N; idim++)
                            D[idim] = displacements[idim][ind];
                    }
                }

                // Swap back positions
                // Bring particles back to the task the eulerian position belongs to
                // The particles are now as they were just with the displacement field assigned
                FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
                part.communicate_particles();
            }

        } // namespace LPT
    }     // namespace COSMOLOGY
} // namespace FML
#endif
