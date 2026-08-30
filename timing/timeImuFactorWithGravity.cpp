/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file timeImuFactorWithGravity.cpp
 * @brief Benchmark the gravity-aware IMU factors next to their plain siblings:
 * preintegration per backend, then error-only and full linearize() cost.
 */

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/CombinedImuFactorWithGravity.h>
#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuFactorWithGravity.h>
#include <gtsam/navigation/LieGroupPreintegration.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>
#include <gtsam/nonlinear/Values.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "internal/TimingUtils.h"

namespace {

using namespace gtsam;
using imuBias::ConstantBias;
using timing::MedianPolicy;
using timing::TimingSummary;

constexpr double kDt = 0.005;
const Vector3 kMeasuredAcceleration{0.1, -0.2, 9.7};
const Vector3 kMeasuredAngularVelocity{0.01, -0.02, 0.015};
volatile double benchmarkSink = 0.0;

const Key kPoseI = Symbol('x', 1), kVelI = Symbol('v', 1);
const Key kPoseJ = Symbol('x', 2), kVelJ = Symbol('v', 2);
const Key kStateI = Symbol('n', 1), kStateJ = Symbol('n', 2);
const Key kBiasI = Symbol('b', 1), kBiasJ = Symbol('b', 2);
const Key kGravityDirection = Symbol('g', 0), kGravityVector = Symbol('g', 1);

std::shared_ptr<PreintegrationParams> makeParams() {
  auto params = PreintegrationParams::MakeSharedU(9.81);
  params->setAccelerometerCovariance(I_3x3 * 1e-4);
  params->setGyroscopeCovariance(I_3x3 * 1e-6);
  params->setIntegrationCovariance(I_3x3 * 1e-8);
  return params;
}

std::shared_ptr<PreintegrationCombinedParams> makeCombinedParams() {
  auto params = PreintegrationCombinedParams::MakeSharedU(9.81);
  params->setAccelerometerCovariance(I_3x3 * 1e-4);
  params->setGyroscopeCovariance(I_3x3 * 1e-6);
  params->setIntegrationCovariance(I_3x3 * 1e-8);
  params->setBiasAccCovariance(I_3x3 * 1e-7);
  params->setBiasOmegaCovariance(I_3x3 * 1e-8);
  return params;
}

// A linearization point away from identity so no Jacobian block is trivial.
Values makeValues() {
  const Pose3 poseI(Rot3::RzRyRx(0.1, -0.2, 0.3), Point3(1.0, 2.0, 3.0));
  const Pose3 poseJ(Rot3::RzRyRx(0.12, -0.18, 0.33), Point3(1.5, 2.1, 3.05));
  const Vector3 velI(0.5, -0.1, 0.05), velJ(0.55, -0.08, 0.02);
  Values values;
  values.insert(kPoseI, poseI);
  values.insert(kVelI, velI);
  values.insert(kPoseJ, poseJ);
  values.insert(kVelJ, velJ);
  values.insert(kStateI, NavState(poseI, velI));
  values.insert(kStateJ, NavState(poseJ, velJ));
  values.insert(kBiasI, ConstantBias(Vector3(0.01, -0.02, 0.03),
                                     Vector3(0.001, 0.002, -0.001)));
  values.insert(kBiasJ, ConstantBias(Vector3(0.011, -0.019, 0.031),
                                     Vector3(0.0011, 0.0019, -0.0012)));
  values.insert(kGravityDirection, Unit3(0.1, -0.2, -1.0));
  values.insert(kGravityVector, Point3(0.4, -0.6, -9.5));
  return values;
}

template <class Pim, class Params>
Pim integrate(const Params& params, size_t samples) {
  Pim pim(params, ConstantBias());
  for (size_t sample = 0; sample < samples; ++sample)
    pim.integrateMeasurement(kMeasuredAcceleration, kMeasuredAngularVelocity,
                             kDt);
  return pim;
}

struct Options {
  size_t samples, calls, warmups, repetitions;
};

template <class Pim, class Params>
double timeIntegration(const Params& params, const Options& options) {
  const auto times = timing::measureMilliseconds(
      [&] {
        const Pim pim = integrate<Pim>(params, options.samples);
        benchmarkSink += pim.deltaTij();
      },
      options.warmups, options.repetitions);
  const TimingSummary summary =
      timing::summarizeSamples(times, MedianPolicy::kUpperMiddle);
  return 1e6 * summary.median / static_cast<double>(options.samples);
}

// Returns {error-only, linearize} in nanoseconds per call.
template <class Factor>
std::pair<double, double> timeFactor(const Factor& factor, const Values& values,
                                     const Options& options) {
  const auto errorTimes = timing::measureMilliseconds(
      [&] {
        for (size_t call = 0; call < options.calls; ++call)
          benchmarkSink += factor.unwhitenedError(values)(0);
      },
      options.warmups, options.repetitions);
  const auto linearizeTimes = timing::measureMilliseconds(
      [&] {
        for (size_t call = 0; call < options.calls; ++call)
          benchmarkSink += factor.linearize(values)->size();
      },
      options.warmups, options.repetitions);
  const double calls = static_cast<double>(options.calls);
  return {1e6 * timing::summarizeSamples(errorTimes, MedianPolicy::kUpperMiddle)
                    .median /
              calls,
          1e6 * timing::summarizeSamples(linearizeTimes,
                                         MedianPolicy::kUpperMiddle)
                    .median /
              calls};
}

void printRow(const std::string& backend, const std::string& factor,
              std::pair<double, double> ns) {
  std::cout << std::left << std::setw(10) << backend << std::setw(38) << factor
            << std::right << std::fixed << std::setprecision(0) << std::setw(12)
            << ns.first << std::setw(14) << ns.second << '\n';
}

template <class Pim>
void benchmarkBackend(const std::string& name, const Options& options) {
  const auto params = makeParams();
  const Pim pim = integrate<Pim>(params, options.samples);
  const Values values = makeValues();
  printRow(name, "ImuFactor",
           timeFactor(ImuFactorT<Pim>(kPoseI, kVelI, kPoseJ, kVelJ, kBiasI, pim),
                      values, options));
  printRow(name, "ImuFactorWithGravity<Unit3>",
           timeFactor(ImuFactorWithGravityT<Pim, Unit3>(
                          kPoseI, kVelI, kPoseJ, kVelJ, kBiasI,
                          kGravityDirection, pim),
                      values, options));
  printRow(name, "ImuFactorWithGravity<Point3>",
           timeFactor(ImuFactorWithGravityT<Pim, Point3>(
                          kPoseI, kVelI, kPoseJ, kVelJ, kBiasI, kGravityVector,
                          pim),
                      values, options));
  printRow(name, "ImuFactor2",
           timeFactor(ImuFactor2T<Pim>(kStateI, kStateJ, kBiasI, pim), values,
                      options));
  printRow(name, "ImuFactor2WithGravity<Unit3>",
           timeFactor(ImuFactor2WithGravityT<Pim, Unit3>(
                          kStateI, kStateJ, kBiasI, kGravityDirection, pim),
                      values, options));
  printRow(name, "ImuFactor2WithGravity<Point3>",
           timeFactor(ImuFactor2WithGravityT<Pim, Point3>(
                          kStateI, kStateJ, kBiasI, kGravityVector, pim),
                      values, options));
}

template <class Pim>
void benchmarkCombinedBackend(const std::string& name, const Options& options) {
  const auto params = makeCombinedParams();
  const Pim pim = integrate<Pim>(params, options.samples);
  const Values values = makeValues();
  printRow(name, "CombinedImuFactor",
           timeFactor(CombinedImuFactorT<Pim>(kPoseI, kVelI, kPoseJ, kVelJ,
                                              kBiasI, kBiasJ, pim),
                      values, options));
  printRow(name, "CombinedImuFactorWithGravity<Unit3>",
           timeFactor(CombinedImuFactorWithGravityT<Pim, Unit3>(
                          kPoseI, kVelI, kPoseJ, kVelJ, kBiasI, kBiasJ,
                          kGravityDirection, pim),
                      values, options));
  printRow(name, "CombinedImuFactorWithGravity<Point3>",
           timeFactor(CombinedImuFactorWithGravityT<Pim, Point3>(
                          kPoseI, kVelI, kPoseJ, kVelJ, kBiasI, kBiasJ,
                          kGravityVector, pim),
                      values, options));
}

}  // namespace

int main(int argc, const char* argv[]) {
  timing::Arguments arguments(argc, argv);
  if (arguments.helpRequested()) {
    std::cout << "Usage: timeImuFactorWithGravity [--samples N] [--calls N] "
                 "[--warmups N] [--repetitions N]\n";
    return 0;
  }
  const Options options{arguments.sizeValue("--samples", 200),
                        arguments.sizeValue("--calls", 2000),
                        arguments.sizeValue("--warmups", 5),
                        arguments.sizeValue("--repetitions", 21)};
  arguments.validateAllConsumed();

  using PT = PreintegratedImuMeasurementsT<TangentPreintegration>;
  using PM = PreintegratedImuMeasurementsT<ManifoldPreintegration>;
  using PL = PreintegratedImuMeasurementsT<LieGroupPreintegration>;
  using PG = PreintegratedImuMeasurementsG;
  using CT = PreintegratedCombinedMeasurementsT<TangentPreintegration>;
  using CM = PreintegratedCombinedMeasurementsT<ManifoldPreintegration>;
  using CL = PreintegratedCombinedMeasurementsT<LieGroupPreintegration>;

  std::cout << "Preintegration (nanoseconds per sample, " << options.samples
            << " samples)\n"
            << std::left << std::setw(10) << "backend" << std::right
            << std::setw(12) << "PIM" << std::setw(16) << "Combined PIM"
            << '\n';
  const auto params = makeParams();
  const auto combinedParams = makeCombinedParams();
  auto printIntegration = [&](const std::string& name, double pim,
                              double combined) {
    std::cout << std::left << std::setw(10) << name << std::right << std::fixed
              << std::setprecision(0) << std::setw(12) << pim << std::setw(16);
    if (combined < 0) std::cout << "-"; else std::cout << combined;
    std::cout << '\n';
  };
  printIntegration("Tangent", timeIntegration<PT>(params, options),
                   timeIntegration<CT>(combinedParams, options));
  printIntegration("Manifold", timeIntegration<PM>(params, options),
                   timeIntegration<CM>(combinedParams, options));
  printIntegration("LieGroup", timeIntegration<PL>(params, options),
                   timeIntegration<CL>(combinedParams, options));
  printIntegration("Galilean", timeIntegration<PG>(params, options), -1.0);

  std::cout << "\nFactor evaluation (nanoseconds per call, " << options.calls
            << " calls)\n"
            << std::left << std::setw(10) << "backend" << std::setw(38)
            << "factor" << std::right << std::setw(12) << "error"
            << std::setw(14) << "linearize" << '\n';
  benchmarkBackend<PT>("Tangent", options);
  benchmarkBackend<PM>("Manifold", options);
  benchmarkBackend<PL>("LieGroup", options);
  benchmarkBackend<PG>("Galilean", options);
  benchmarkCombinedBackend<CT>("Tangent", options);
  benchmarkCombinedBackend<CM>("Manifold", options);
  benchmarkCombinedBackend<CL>("LieGroup", options);
  return benchmarkSink == 0.0 ? 1 : 0;
}
