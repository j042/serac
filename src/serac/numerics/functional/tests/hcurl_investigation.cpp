// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

static constexpr double a = 1.7;
static constexpr double b = 2.1;

template <int dim>
struct hcurl_qfunction {
  template <typename x_t, typename vector_potential_t>
  SERAC_HOST_DEVICE auto operator()(x_t x, vector_potential_t vector_potential) const
  {
    auto [A, curl_A] = vector_potential;
    auto J_term      = a * A - tensor<double, dim>{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]};
    auto H_term      = b * curl_A;
    return serac::tuple{J_term, H_term};
  }
};

// this test sets up part of a toy "magnetic diffusion" problem where the residual includes contributions
// from a vector-potential-proportional J and an isotropically linear H
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, Hcurl<p> test, Hcurl<p> trial, Dimension<dim>)
{
  auto                        fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParBilinearForm B(&fespace);

  mfem::ConstantCoefficient a_coef(a);
  B.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(a_coef));

  mfem::ConstantCoefficient b_coef(b);
  B.AddDomainIntegrator(new mfem::CurlCurlIntegrator(b_coef));
  B.Assemble(0);
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J_mfem(B.ParallelAssemble());

  mfem::ParLinearForm             f(&fespace);
  mfem::VectorFunctionCoefficient load_func(dim, [&](const mfem::Vector& coords, mfem::Vector& output) {
    double x  = coords(0);
    double y  = coords(1);
    output    = 0.0;
    output(0) = 10 * x * y;
    output(1) = -5 * (x - y) * y;
  });

  f.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddDomainIntegral(Dimension<dim>{}, hcurl_qfunction<dim>{}, mesh);

  mfem::Vector r1 = (*J_mfem) * U - (*F);
  mfem::Vector r2 = residual(U);

  auto [r, drdU] = residual(differentiate_wrt(U));

  std::unique_ptr<mfem::HypreParMatrix> J_func = assemble(drdU);

  mfem::Vector g1 = (*J_mfem) * U;
  mfem::Vector g2 = drdU * U;
  mfem::Vector g3 = (*J_func) * U;

  residual.inspect();

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g3||: " << g3.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  std::cout << "||g1-g3||/||g1||: " << mfem::Vector(g1 - g3).Norml2() / g1.Norml2() << std::endl;
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  //std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  //std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/square.mesh";
  //mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), 0, 0);
  //mesh2D->ExchangeFaceNbrData();
  //functional_test(*mesh2D, Hcurl<2>{}, Hcurl<2>{}, Dimension<2>{});

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/onehex.mesh";
  //std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), 0, 0);
  mesh3D->ExchangeFaceNbrData();
  functional_test(*mesh3D, Hcurl<2>{}, Hcurl<2>{}, Dimension<3>{});

  MPI_Finalize();
}
