
#include <random>

#include "serac/physics/utilities/functional/tuple.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"

using namespace serac;

auto random_real = [](auto...) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
};

int main() {

    auto x = make_tensor<3>(random_real);

    auto phi = random_real();
    auto grad_phi = make_tensor<3>(random_real);
    serac::tuple temperature{phi, grad_phi};

    auto u = make_tensor<3>(random_real);
    auto grad_u = make_tensor<3, 3>(random_real);
    serac::tuple displacement{u, grad_u};

    auto f = [](auto x, auto temperature, auto displacement) {
        auto [phi, grad_phi] = temperature;
        auto [u, grad_u] = displacement;

        auto source = x[0] + u[1] * phi;
        auto flux = grad_u * grad_phi;

        return tuple{source, flux};
    };

    auto [source, flux] = f(x, temperature, displacement);

    [[maybe_unused]] auto args = make_dual(temperature, displacement);
    // auto output = f(x, std::get<0>(args), std::get<1>(args));
    
    // output: serac::tuple <
    //   dual< serac::tuple < serac::tuple < > , serac::tuple < > > >, 
    //   tensor < dual< serac::tuple < serac::tuple < >, serac::tuple < > > >, 3 > 
    // > 
    // 




    std::cout << source << ", " << flux << std::endl;

}
