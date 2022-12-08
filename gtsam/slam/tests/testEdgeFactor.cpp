#include <gtsam/slam/EdgeFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <CppUnitLite/TestHarness.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

std::mt19937 mt(42);
size_t max_points = 40;
double max_edge_len = 0.5;

void getOptimizedError(const NonlinearFactorGraph & graph, const Values& initial, const Values& ground_truth, Pose3 & p1_error, Pose3 & p2_error)
{
  ISAM2 isam2;
  isam2.update(graph, initial);
  Values result = isam2.calculateEstimate();
  Pose3 optimized_p1 = result.at<Pose3>(X(1));
  Pose3 optimized_p2 = result.at<Pose3>(X(2));
  p1_error = optimized_p1.between(ground_truth.at<Pose3>(X(1)));
  p2_error = optimized_p2.between(ground_truth.at<Pose3>(X(2)));
}

// Random points random poses
TEST(EdgeFactor2, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);

  for (size_t iter = 1; iter < max_points; ++iter)
  {
    Point3 j = Point3::Random();
    Vector3 dv = Vector3::Random();
    dv.normalize();
    double edge_len = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l = j + edge_len * dv;
    float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_in_p1 = j + lambda * (l - j);
    Point3 i_in_p2 = p2.transformTo(p1.transformFrom(i_in_p1));
    // Add noise
    Point3 i = i_in_p2 + Point3::Random().normalized() * 0.001;

    EdgeFactor2 factor(j, l, X(1), i, X(2), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
    graph.add(factor);
  }

  Pose3 p1_error, p2_error;
  getOptimizedError(graph, values, values, p1_error, p2_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
}

void getOptimizedError(const NonlinearFactorGraph & graph, const Values& initial, const Values& ground_truth, Pose3 & p1_error, Pose3 & p2_error, Pose3 & p3_error)
{
  ISAM2 isam2;
  isam2.update(graph, initial);
  Values result = isam2.calculateEstimate();
  Pose3 optimized_p1 = result.at<Pose3>(X(1));
  Pose3 optimized_p2 = result.at<Pose3>(X(2));
  Pose3 optimized_p3 = result.at<Pose3>(X(3));
  p1_error = optimized_p1.between(ground_truth.at<Pose3>(X(1)));
  p2_error = optimized_p2.between(ground_truth.at<Pose3>(X(2)));
  p3_error = optimized_p3.between(ground_truth.at<Pose3>(X(3)));
}

// Random points random poses
TEST(EdgeFactor3, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));
  graph.addPrior<Pose3>(X(2), p2, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);

  for (size_t iter = 1; iter < max_points; ++iter)
  {
    Point3 j_in_w = Point3::Random();
    Vector3 dv = Vector3::Random();
    dv.normalize();
    double edge_len = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l_in_w = j_in_w + edge_len * dv;
    Point3 j = p1.transformTo(j_in_w);
    Point3 l = p2.transformTo(l_in_w);
    float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_in_w = j_in_w + lambda * (l_in_w - j_in_w);
    Point3 i_in_p3 = p3.transformTo(i_in_w);
    // Add noise
    Point3 i = i_in_p3 + Point3::Random().normalized() * 0.01;

    EdgeFactor3 factor(j, X(1), l, X(2), i, X(3), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-4);
    graph.add(factor);
  }

  Pose3 p1_error, p2_error, p3_error;
  getOptimizedError(graph, values, values, p1_error, p2_error, p3_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
}

// // Random points same pose
// TEST(EdgeFactor2, JacobiansAndOpt2)
// {
//   Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
//   Pose3 p2 = p1;

//   NonlinearFactorGraph graph;
//   graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);

//   for (size_t iter = 1; iter < max_points; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 l = Point3::Random();
//     Point3 i = Point3::Random();

//     EdgeFactor2 factor(j, l, X(1), i, X(2), noiseModel::Unit::Create(3));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
//     graph.add(factor);
//   }

//   Pose3 p1_error, p2_error;
//   getOptimizedError(graph, values, values, p1_error, p2_error);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
// }

// // Same points same pose
// TEST(EdgeFactor2, JacobiansAndOpt3)
// {
//   Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
//   Pose3 p2 = p1;

//   NonlinearFactorGraph graph;
//   graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);

//   for (size_t iter = 1; iter < max_points; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 l = l;
//     Point3 i = i;

//     EdgeFactor2 factor(j, l, X(1), i, X(2), noiseModel::Unit::Create(3));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
//     graph.add(factor);
//   }

//   Pose3 p1_error, p2_error;
//   getOptimizedError(graph, values, values, p1_error, p2_error);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
// }

// void getOptimizedError(const NonlinearFactorGraph & graph, const Values& initial, const Values& ground_truth, Pose3 & p1_error, Pose3 & p2_error, Pose3 & p3_error)
// {
//   ISAM2 isam2;
//   isam2.update(graph, initial);
//   Values result = isam2.calculateEstimate();
//   Pose3 optimized_p1 = result.at<Pose3>(X(1));
//   Pose3 optimized_p2 = result.at<Pose3>(X(2));
//   Pose3 optimized_p3 = result.at<Pose3>(X(3));
//   p1_error = optimized_p1.between(ground_truth.at<Pose3>(X(1)));
//   p2_error = optimized_p2.between(ground_truth.at<Pose3>(X(2)));
//   p3_error = optimized_p3.between(ground_truth.at<Pose3>(X(3)));
// }

// // Random points random poses
// TEST(EdgeFactor3, JacobiansAndOpt1)
// {
//   Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
//   Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
//   Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());

//   NonlinearFactorGraph graph;
//   graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   values.insert(X(3), p3);

//   for (size_t iter = 1; iter < max_points; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 l = Point3::Random();
//     Point3 i = Point3::Random();

//     EdgeFactor3 factor(j, X(1), l, X(2), i, X(3), noiseModel::Unit::Create(3));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
//     graph.add(factor);
//   }

//   Pose3 p1_error, p2_error, p3_error;
//   getOptimizedError(graph, values, values, p1_error, p2_error, p3_error);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
// }

// void testEdgeFactor3(const Pose3 p1, const Pose3 p2, const Pose3 p3, const Key k1, const Key k2, const Key k3, const size_t num_points, const float pertubation_rot)
// {
//   NonlinearFactorGraph graph;
//   PriorFactor<Pose3> prior1(k1, p1, noiseModel::Diagonal::Sigmas(Vector6::Constant(0.1)));
//   graph.add(prior1);

//   Pose3 p1_noise(Rot3::Random(mt, pertubation_rot), Point3::Random());
//   Pose3 p2_noise(Rot3::Random(mt, pertubation_rot), Point3::Random());
//   Pose3 p3_noise(Rot3::Random(mt, pertubation_rot), Point3::Random());

//   Pose3 p1_perturbed = p1.compose(p1_noise);
//   Pose3 p2_perturbed = p2.compose(p2_noise);
//   Pose3 p3_perturbed = p3.compose(p3_noise);
  
//   Values values;
//   values.insert(k1, p1_perturbed);
//   if (k1 != k2)
//     values.insert(k2, p2_perturbed);
//   values.insert(k3, p3_perturbed);

//   for (size_t iter = 0; iter < num_points; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 l = Point3::Random();
//     float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//     Point3 i_in_w = p1.transformFrom(j) * lambda + p2.transformFrom(l) * (1 - lambda);
//     Point3 i = p3.transformTo(i_in_w);

//     EdgeFactor3 factor(j, k1, l, k2, i, k3, noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
//     graph.add(factor);
//   }

//   // Test with ISAM2
//   ISAM2 isam2;
//   isam2.update(graph, values);
//   Values result = isam2.calculateEstimate();
//   Pose3 optimized_p1 = result.at<Pose3>(X(1));
//   Pose3 optimized_p2 = result.at<Pose3>(X(2));
//   Pose3 optimized_p3 = result.at<Pose3>(X(3));
//   Pose3 p1_error = optimized_p1.between(p1);
//   Pose3 p2_error = optimized_p2.between(p2);
//   Pose3 p3_error = optimized_p3.between(p3);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
// }

// TEST(EdgeFactor3, Jacobians2)
// {
//   Pose3 p1;
//   Pose3 p2;
//   Pose3 p3;
//   testEdgeFactor3(p1, p2, p3, X(1), X(2), X(3), 100, 0.1);
// }

// TEST(EdgeFactor3, JacobiansSameKeySamePoseOffLine)
// {
//   Pose3 p1;
//   Pose3 p2;
//   Point3 j(1, 2, 3);
//   Point3 l(1, 4, 1);
//   float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//   Point3 i = j + lambda * (l - j) + Point3(1, 1, 1);

//   EdgeFactor3 factor(j, X(1), l, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }

// TEST(EdgeFactor3, JacobiansSameKeyDiffPoseOnLine)
// {
//   Pose3 p1;
//   Pose3 p2(Rot3::Random(mt, M_PI/20), Point3(1, 2, 3));
//   Point3 j(1, 2, 3);
//   Point3 l(1, 4, 1);
//   float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//   Point3 i = j + lambda * (l - j);

//   EdgeFactor3 factor(j, X(1), l, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }

// TEST(EdgeFactor3, JacobiansSameKeyDiffPoseOffLine)
// {
//   Pose3 p1;
//   Pose3 p2(Rot3::Random(mt, M_PI/20), Point3(1, 2, 3));
//   Point3 j(1, 2, 3);
//   Point3 l(1, 4, 1);
//   float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//   Point3 i = j + lambda * (l - j) + Point3(1, 1, 1);

//   EdgeFactor3 factor(j, X(1), l, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }

// TEST(EdgeFactor3, JacobiansDiffKeySamePoseOnLine)
// {
//   Pose3 p1;
//   Pose3 p2;
//   Point3 j(1, 2, 3);
//   Point3 l(1, 4, 1);
//   float lambda = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//   Point3 i = j + lambda * (l - j);

//   EdgeFactor3 factor(j, X(1), l, X(2), i, X(3), noiseModel::Diagonal::Sigmas(Vector3(1, 1, 1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   values.insert(X(3), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }



// TEST(EdgeFactor2, ErrorAndJacobians)
// {
//   Pose3 p1(Rot3::Random(mt), Point3::Random());
//   Pose3 p2(Rot3::Random(mt), Point3::Random());

//   NonlinearFactorGraph graph;
//   PriorFactor<Pose3> prior1(X(1), p1, noiseModel::Isotropic::Sigma(6, 0.0001));
//   graph.add(prior1);

//   Pose3 p1_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());
//   Pose3 p2_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());

//   Pose3 p1_perturbed = p1.compose(p1_noise);
//   Pose3 p2_perturbed = p2.compose(p2_noise);

//   Values values;
//   values.insert(X(1), p1_perturbed);
//   values.insert(X(2), p2_perturbed);

//   for (size_t iter = 0; iter < 4; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 i = p2.transformTo(p1.transformFrom(j));

//     EdgeFactor2 factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1)));

//     EXPECT(assert_equal(factor.evaluateError(p1, p2), Vector3::Constant(0.0), 1e-5));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);

//     graph.add(factor);
//   }

//   // Test with ISAM2
//   ISAM2 isam2;
//   isam2.update(graph, values);
//   Values result = isam2.calculateEstimate();
//   Pose3 optimized_p1 = result.at<Pose3>(X(1));
//   Pose3 optimized_p2 = result.at<Pose3>(X(2));
//   Pose3 p1_error = optimized_p1.between(p1);
//   Pose3 p2_error = optimized_p2.between(p2);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
// }

// // This test is disabled because it is not working yet
// // TEST(PointFactorS, JacobiansIdenticalPoints)
// // {
// //   Pose3 p1;
// //   Pose3 p2;
// //   Point3 j(1, 2, 3);
// //   Point3 i(1, 2, 3);

// //   PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(1)));
// //   Values values;
// //   values.insert(X(1), p1);
// //   values.insert(X(2), p2);
// //   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// // }

// TEST(PointFactorS, JacobiansNonIdenticalPoints)
// {
//   Pose3 p1;
//   Pose3 p2;
//   Point3 j(1, 2, 3.1);
//   Point3 i(1, 2, 3);

//   PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(1)));
//   Values values;
//   values.insert(X(1), p1);
//   values.insert(X(2), p2);
//   EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
// }

// TEST(PointFactorS, ErrorAndJacobians)
// {
//   Pose3 p1(Rot3::Random(mt), Point3::Random());
//   Pose3 p2(Rot3::Random(mt), Point3::Random());

//   p1.print("p1");
//   p2.print("p2");

//   NonlinearFactorGraph graph;
//   PriorFactor<Pose3> prior1(X(1), p1, noiseModel::Isotropic::Sigma(6, 0.001));
//   graph.add(prior1);

//   Pose3 p1_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());
//   Pose3 p2_noise(Rot3::Random(mt, M_PI/20), Point3::Zero());

//   Pose3 p1_perturbed = p1.compose(p1_noise);
//   Pose3 p2_perturbed = p2.compose(p2_noise);

//   Values values;
//   values.insert(X(1), p1_perturbed);
//   values.insert(X(2), p2_perturbed);

//   for (size_t iter = 0; iter < 10; ++iter)
//   {
//     Point3 j = Point3::Random();
//     Point3 i = p2.transformTo(p1.transformFrom(j));

//     PointFactorS factor(j, X(1), i, X(2), noiseModel::Diagonal::Sigmas(Vector1(0.1)));

//     EXPECT(assert_equal(Vector1(0.0), factor.evaluateError(p1, p2), 1e-5));
//     EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);

//     graph.add(factor);
//   }

//   // Test with ISAM2
//   ISAM2 isam2;
//   isam2.update(graph, values);
//   Values result = isam2.calculateEstimate();
//   Pose3 optimized_p1 = result.at<Pose3>(X(1));
//   Pose3 optimized_p2 = result.at<Pose3>(X(2));
//   Pose3 p1_error = optimized_p1.between(p1);
//   Pose3 p2_error = optimized_p2.between(p2);
//   EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
//   EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
//   EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
// }

int main()
{
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}