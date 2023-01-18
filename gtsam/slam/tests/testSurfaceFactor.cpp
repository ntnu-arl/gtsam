#include <gtsam/slam/SurfaceFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <CppUnitLite/TestHarness.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

std::mt19937 mt(4);
double max_edge_len = 0.5;
double min_edge_len = 0.1;

void getOptimizedError(const NonlinearFactorGraph &graph, const Values &initial, const Values &ground_truth, Values &result, Pose3 &p1_error, Pose3 &p2_error)
{
  ISAM2 isam2;
  isam2.update(graph, initial);
  result = isam2.calculateEstimate();
  Pose3 optimized_p1 = result.at<Pose3>(X(1));
  Pose3 optimized_p2 = result.at<Pose3>(X(2));
  p1_error = optimized_p1.between(ground_truth.at<Pose3>(X(1)));
  p2_error = optimized_p2.between(ground_truth.at<Pose3>(X(2)));
}

void getOptimizedError(const NonlinearFactorGraph &graph, const Values &initial, const Values &ground_truth, Values &result, Pose3 &p1_error, Pose3 &p2_error, Pose3 &p3_error)
{
  getOptimizedError(graph, initial, ground_truth, result, p1_error, p2_error);
  Pose3 optimized_p3 = result.at<Pose3>(X(3));
  p3_error = optimized_p3.between(ground_truth.at<Pose3>(X(3)));
}

void getOptimizedError(const NonlinearFactorGraph &graph, const Values &initial, const Values &ground_truth, Values &result, Pose3 &p1_error, Pose3 &p2_error, Pose3 &p3_error, Pose3 & p4_error)
{
  getOptimizedError(graph, initial, ground_truth, result, p1_error, p2_error, p3_error);
  Pose3 optimized_p4 = result.at<Pose3>(X(4));
  p4_error = optimized_p4.between(ground_truth.at<Pose3>(X(4)));
}

// Random points random poses
TEST(SurfaceFactor2, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);

  for (size_t iter = 0; iter < 10; ++iter)
  {
    Point3 j_w = Point3::Random();
    Vector3 dv1_w = Vector3::Random();
    dv1_w.normalize();
    double edge_len1 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l_w = j_w + edge_len1 * dv1_w;
    Vector3 dv2_w = Vector3::Random();
    dv2_w.normalize();
    double edge_len2 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 m_w = j_w + edge_len2 * dv2_w;
    float lambda1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float lambda2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_w = j_w + lambda1 * dv1_w + lambda2 * dv2_w;
    Point3 i = p2.transformTo(i_w) + Point3::Random().normalized() * 0.00001;

    // Reorder points to make sure that m_w is the furthest point
    if ((i_w - j_w).norm() > (i_w - m_w).norm())
      std::swap(j_w, m_w);
    if ((i_w - l_w).norm() > (i_w - m_w).norm())
      std::swap(l_w, m_w);

    // Reorder j_w and l_w to make sure that j_w is the closest point - This is not a requirement as the factor is designed. But this should be tested if jacobians ever start becoming wrong.
    // if ((i_w - j_w).norm() > (i_w - l_w).norm())
    //   std::swap(j_w, l_w);

    // Convert points j, l and m to their local frames
    Point3 j = p1.transformTo(j_w);
    Point3 l = p1.transformTo(l_w);
    Point3 m = p1.transformTo(m_w);

    SurfaceFactor2 factor(j, l, m, X(1), i, X(2), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
    graph.add(factor);
  }

  Values result;
  Pose3 p1_error, p2_error;
  getOptimizedError(graph, values, values, result, p1_error, p2_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
}

// Random points random poses
TEST(SurfaceFactor3JM, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);

  for (size_t iter = 0; iter < 10; ++iter)
  {
    Point3 j_w = Point3::Random();
    Vector3 dv1_w = Vector3::Random();
    dv1_w.normalize();
    double edge_len1 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l_w = j_w + edge_len1 * dv1_w;
    Vector3 dv2_w = Vector3::Random();
    dv2_w.normalize();
    double edge_len2 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 m_w = j_w + edge_len2 * dv2_w;
    float lambda1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float lambda2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_w = j_w + lambda1 * dv1_w + lambda2 * dv2_w;
    Point3 i = p3.transformTo(i_w) + Point3::Random().normalized() * 0.00001;

    // Reorder points to make sure that m_w is the furthest point
    if ((i_w - j_w).norm() > (i_w - m_w).norm())
      std::swap(j_w, m_w);
    if ((i_w - l_w).norm() > (i_w - m_w).norm())
      std::swap(l_w, m_w);

    // Reorder j_w and l_w to make sure that j_w is the closest point - This is not a requirement as the factor is designed. But this should be tested if jacobians ever start becoming wrong.
    // if ((i_w - j_w).norm() > (i_w - l_w).norm())
    //   std::swap(j_w, l_w);

    // Convert points j, l and m to their local frames
    Point3 j = p1.transformTo(j_w);
    Point3 l = p2.transformTo(l_w);
    Point3 m = p1.transformTo(m_w);

    SurfaceFactor3JM factor(j, X(1), l, X(2), m, i, X(3), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
    graph.add(factor);
  }

  Values result;
  Pose3 p1_error, p2_error, p3_error;
  getOptimizedError(graph, values, values, result, p1_error, p2_error, p3_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
}

// Random points random poses
TEST(SurfaceFactor3LM, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);

  for (size_t iter = 0; iter < 10; ++iter)
  {
    Point3 j_w = Point3::Random();
    Vector3 dv1_w = Vector3::Random();
    dv1_w.normalize();
    double edge_len1 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l_w = j_w + edge_len1 * dv1_w;
    Vector3 dv2_w = Vector3::Random();
    dv2_w.normalize();
    double edge_len2 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 m_w = j_w + edge_len2 * dv2_w;
    float lambda1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float lambda2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_w = j_w + lambda1 * dv1_w + lambda2 * dv2_w;
    Point3 i = p3.transformTo(i_w) + Point3::Random().normalized() * 0.00001;

    // Reorder points to make sure that m_w is the furthest point
    if ((i_w - j_w).norm() > (i_w - m_w).norm())
      std::swap(j_w, m_w);
    if ((i_w - l_w).norm() > (i_w - m_w).norm())
      std::swap(l_w, m_w);

    // Reorder j_w and l_w to make sure that j_w is the closest point - This is not a requirement as the factor is designed. But this should be tested if jacobians ever start becoming wrong.
    // if ((i_w - j_w).norm() > (i_w - l_w).norm())
    //   std::swap(j_w, l_w);

    // Convert points j, l and m to their local frames
    Point3 j = p1.transformTo(j_w);
    Point3 l = p2.transformTo(l_w);
    Point3 m = p2.transformTo(m_w);

    SurfaceFactor3LM factor(j, X(1), l, X(2), m, i, X(3), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
    graph.add(factor);
  }

  Values result;
  Pose3 p1_error, p2_error, p3_error;
  getOptimizedError(graph, values, values, result, p1_error, p2_error, p3_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
}

// Random points random poses
TEST(SurfaceFactor3JL, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);

  for (size_t iter = 0; iter < 10; ++iter)
  {
    Point3 j_w = Point3::Random();
    Vector3 dv1_w = Vector3::Random();
    dv1_w.normalize();
    double edge_len1 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 l_w = j_w + edge_len1 * dv1_w;
    Vector3 dv2_w = Vector3::Random();
    dv2_w.normalize();
    double edge_len2 = rand() / static_cast<double>(RAND_MAX) * max_edge_len;
    Point3 m_w = j_w + edge_len2 * dv2_w;
    float lambda1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float lambda2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Point3 i_w = j_w + lambda1 * dv1_w + lambda2 * dv2_w;
    Point3 i = p3.transformTo(i_w) + Point3::Random().normalized() * 0.00001;

    // Reorder points to make sure that m_w is the furthest point
    if ((i_w - j_w).norm() > (i_w - m_w).norm())
      std::swap(j_w, m_w);
    if ((i_w - l_w).norm() > (i_w - m_w).norm())
      std::swap(l_w, m_w);

    // Reorder j_w and l_w to make sure that j_w is the closest point - This is not a requirement as the factor is designed. But this should be tested if jacobians ever start becoming wrong.
    // if ((i_w - j_w).norm() > (i_w - l_w).norm())
    //   std::swap(j_w, l_w);

    // Convert points j, l and m to their local frames
    Point3 j = p1.transformTo(j_w);
    Point3 l = p1.transformTo(l_w);
    Point3 m = p2.transformTo(m_w);

    SurfaceFactor3JL factor(j, X(1), l, m, X(2), i, X(3), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-3);
    graph.add(factor);
  }

  Values result;
  Pose3 p1_error, p2_error, p3_error;
  getOptimizedError(graph, values, values, result, p1_error, p2_error, p3_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
}

// Random points random poses
TEST(SurfaceFactor4, JacobiansAndOpt1)
{
  Pose3 p1(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p2(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p3(Rot3::Random(mt, M_PI/20.0), Point3::Random());
  Pose3 p4(Rot3::Random(mt, M_PI/20.0), Point3::Random());

  NonlinearFactorGraph graph;
  graph.addPrior<Pose3>(X(1), p1, noiseModel::Unit::Create(6));

  Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);
  values.insert(X(4), p4);

  for (size_t iter = 0; iter < 15; ++iter)
  {
    Point3 j_w = Point3::Random();
    Vector3 dv1_w = Vector3::Random();
    dv1_w.normalize();
    double edge_len1 = std::max(rand() * max_edge_len / static_cast<double>(RAND_MAX), min_edge_len);
    Point3 l_w = j_w + edge_len1 * dv1_w;
    Vector3 dv2_w = Vector3::Random();
    dv2_w.normalize();
    double edge_len2 = std::max(rand() * max_edge_len / static_cast<double>(RAND_MAX), min_edge_len);
    Point3 m_w = j_w + edge_len2 * dv2_w;
    float lambda1 = static_cast<float>(rand()) * edge_len1 / static_cast<float>(RAND_MAX);
    float lambda2 = static_cast<float>(rand()) * edge_len2 / static_cast<float>(RAND_MAX);
    Point3 i_w = j_w + lambda1 * dv1_w + lambda2 * dv2_w;
    Point3 i = p4.transformTo(i_w) + Point3::Random().normalized() * 0.00001;

    // Reorder points to make sure that m_w is the furthest point
    if ((i_w - j_w).norm() > (i_w - m_w).norm())
      std::swap(j_w, m_w);
    if ((i_w - l_w).norm() > (i_w - m_w).norm())
      std::swap(l_w, m_w);

    // Reorder j_w and l_w to make sure that j_w is the closest point - This is not a requirement as the factor is designed. But this should be tested if jacobians ever start becoming wrong.
    // if ((i_w - j_w).norm() > (i_w - l_w).norm())
    //   std::swap(j_w, l_w);

    // Convert points j, l and m to their local frames
    Point3 j = p1.transformTo(j_w);
    Point3 l = p2.transformTo(l_w);
    Point3 m = p3.transformTo(m_w);

    SurfaceFactor4 factor(j, X(1), l, X(2), m, X(3), i, X(4), noiseModel::Unit::Create(3));
    EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-5, 1e-5);
    graph.add(factor);
  }

  Values result;
  Pose3 p1_error, p2_error, p3_error, p4_error;
  getOptimizedError(graph, values, values, result, p1_error, p2_error, p3_error, p4_error);
  EXPECT(assert_equal(0.0, p1_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p2_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p3_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p4_error.translation().norm(), 1e-2));
  EXPECT(assert_equal(0.0, p1_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p2_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p3_error.rotation().axisAngle().second, 1e-2));
  EXPECT(assert_equal(0.0, p4_error.rotation().axisAngle().second, 1e-2));
}

int main()
{
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}