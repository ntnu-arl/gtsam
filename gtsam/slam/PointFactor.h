#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam
{

class GTSAM_EXPORT PointFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
  gtsam::Point3 j_;  // In the frame of the first key
  gtsam::Point3 i_;  // In the frame of the second key

public:
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;
  typedef PointFactor This;
  typedef boost::shared_ptr<This> shared_ptr;

  PointFactor(
    const gtsam::Point3 & j, gtsam::Key k1, const gtsam::Point3 & i, gtsam::Key k2,
    const gtsam::SharedNoiseModel & model)
  : Base(model, k1, k2), j_(j), i_(i)
  {
  }

  virtual ~PointFactor() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & p1, const gtsam::Pose3 & p2,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override
  {
    gtsam::Point3 transformed_j = p1.transformFrom(j_, H1);
    gtsam::Point3 transformed_i = p2.transformFrom(i_, H2);
    gtsam::Vector3 residual(
      transformed_i.x() - transformed_j.x(), transformed_i.y() - transformed_j.y(),
      transformed_i.z() - transformed_j.z());
    if (H1) {
      *H1 = -*H1;
    }
    return residual;
  }
};

class GTSAM_EXPORT PointFactorS : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
  gtsam::Point3 j_;  // In the frame of the first key
  gtsam::Point3 i_;  // In the frame of the second key

public:
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;
  typedef PointFactorS This;
  typedef boost::shared_ptr<This> shared_ptr;

  PointFactorS(
    const gtsam::Point3 & j, gtsam::Key k1, const gtsam::Point3 & i, gtsam::Key k2,
    const gtsam::SharedNoiseModel & model)
  : Base(model, k1, k2), j_(j), i_(i)
  {
  }

  virtual ~PointFactorS() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & p1, const gtsam::Pose3 & p2,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override
  {
    gtsam::Point3 transformed_j = p1.transformFrom(j_, H1);
    gtsam::Point3 transformed_i = p2.transformFrom(i_, H2);
    gtsam::Vector3 residual_v(
      transformed_i.x() - transformed_j.x(), transformed_i.y() - transformed_j.y(),
      transformed_i.z() - transformed_j.z());
    gtsam::Vector1 residual(residual_v.norm());
    gtsam::Vector3 residual_v_normalized;
    if (H1 || H2)
    {
      // There is a risk of a divide by zero here
      // It causes the jacobians to be NaN 
      // TODO: Fix this
      residual_v_normalized = residual_v / residual(0);
    }
    if (H1) {
      *H1 = -residual_v_normalized.transpose() * *H1;
    }
    if (H2) {
      *H2 = residual_v_normalized.transpose() * *H2;
    }
    return residual;
  }
};
}  // namespace gtsam
