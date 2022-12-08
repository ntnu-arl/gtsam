#include <gtsam/slam/PointFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <CppUnitLite/TestHarness.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

std::mt19937 mt(42);

TEST(PointFactor, Jacobians)
{
  Pose3 p1;
  Pose3 p2(Rot3(), Point3(1, 2, 3));
  Point3 j(1, 2, 3);
  Point3 i(3, 4, 8);

  PointFactor factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
}

TEST(PointFactor, JacobiansIdenticalPoints)
{
  Pose3 p1;
  Pose3 p2;
  Point3 j(1, 2, 3);
  Point3 i(1, 2, 3);

  PointFactor factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
}


TEST(PointFactor, ErrorAndJacobians)
{
  Pose3 p1(Rot3::Random(mt), Point3::Random());
  Pose3 p2(Rot3::Random(mt), Point3::Random());

  NonlinearFactorGraph graph;
  PriorFactor<Pose3> prior1(X(1), p1, noiseModel::Isotropic::Sigma(6, 0.0001));
  graph.add(prior1);

  Pose3 p1_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());
  Pose3 p2_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());

  Pose3 p1_perturbed = p1.compose(p1_noise);
  Pose3 p2_perturbed = p2.compose(p2_noise);

  Values values;
  values.insert(X(1), p1_perturbed);
  values.insert(X(2), p2_perturbed);

  for (size_t iter = 0; iter < 4; ++iter)
  {
    Point3 j = Point3::Random();
    Point3 i = p2.transformTo(p1.transformFrom(j));

    PointFactor factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1)));

    EXPECT(assert_equal(factor.evaluateError(p1, p2), Vector3::Constant(0.0), 1e-5));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);

    graph.add(factor);
  }

  // Test with ISAM2
  ISAM2 isam2;
  isam2.update(graph, values);
  Values result = isam2.calculateEstimate();
  Pose3 optimized_p1 = result.at<Pose3>(X(1));
  Pose3 optimized_p2 = result.at<Pose3>(X(2));
  Pose3 p1_error = optimized_p1.between(p1);
  Pose3 p2_error = optimized_p2.between(p2);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
}

// This test is disabled because it is not working yet
// TEST(PointFactorS, JacobiansIdenticalPoints)
// {
//   Pose3 p1;
//   Pose3 p2;
//   Point3 j(1, 2, 3);
//   Point3 i(1, 2, 3);

//   PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }

TEST(PointFactorS, JacobiansNonIdenticalPoints)
{
  Pose3 p1;
  Pose3 p2;
  Point3 j(1, 2, 3.1);
  Point3 i(1, 2, 3);

  PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(1)));
  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
}

TEST(PointFactorS, ErrorAndJacobians)
{
  Pose3 p1(Rot3::Random(mt), Point3::Random());
  Pose3 p2(Rot3::Random(mt), Point3::Random());

  p1.print("p1");
  p2.print("p2");

  NonlinearFactorGraph graph;
  PriorFactor<Pose3> prior1(X(1), p1, noiseModel::Isotropic::Sigma(6, 0.001));
  graph.add(prior1);

  Pose3 p1_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());
  Pose3 p2_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());

  Pose3 p1_perturbed = p1.compose(p1_noise);
  Pose3 p2_perturbed = p2.compose(p2_noise);

  Values values;
  values.insert(X(1), p1_perturbed);
  values.insert(X(2), p2_perturbed);

  for (size_t iter = 0; iter < 10; ++iter)
  {
    Point3 j = Point3::Random();
    Point3 i = p2.transformTo(p1.transformFrom(j));

    PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(0.1)));

    EXPECT(assert_equal(Vector1(0.0), factor.evaluateError(p1, p2), 1e-5));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);

    graph.add(factor);
  }

  // Test with ISAM2
  ISAM2 isam2;
  isam2.update(graph, values);
  Values result = isam2.calculateEstimate();
  Pose3 optimized_p1 = result.at<Pose3>(X(1));
  Pose3 optimized_p2 = result.at<Pose3>(X(2));
  Pose3 p1_error = optimized_p1.between(p1);
  Pose3 p2_error = optimized_p2.between(p2);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
}

int main()
{
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}