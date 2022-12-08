#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam
{
class GTSAM_EXPORT EdgeFactor2 : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
  gtsam::Point3 j0_;  // In the frame of the first key
  gtsam::Point3 l0_;  // In the frame of the first key
  gtsam::Point3 i1_;  // In the frame of the second key

public:
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;
  typedef EdgeFactor2 This;
  typedef boost::shared_ptr<This> shared_ptr;

  EdgeFactor2(
    const gtsam::Point3 & j0, const gtsam::Point3 & l0, gtsam::Key k0, const gtsam::Point3 & i1,
    gtsam::Key k1, const gtsam::SharedNoiseModel & model)
  : Base(model, k0, k1), j0_(j0), l0_(l0), i1_(i1)
  {
  }

  virtual ~EdgeFactor2() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & p0, const gtsam::Pose3 & p1,
    boost::optional<gtsam::Matrix &> H0 = boost::none,
    boost::optional<gtsam::Matrix &> H1 = boost::none) const override
  {
    gtsam::Point3 i_w = p1.transformFrom(i1_);
    gtsam::Point3 l_w = p0.transformFrom(l0_);
    gtsam::Vector3 dv_0 = (j0_ - l0_) / (j0_ - l0_).norm();

    gtsam::Matrix33 Rw0 = p0.rotation().matrix();
    gtsam::Matrix33 Rw1 = p1.rotation().matrix();
    gtsam::Vector3 dv_w = Rw0 * dv_0;

    gtsam::Vector3 residual = (i_w - l_w).cross(dv_w);

    if (H0) {
      gtsam::Matrix36 temp1;
      temp1.leftCols<3>() = -Rw0 * gtsam::skewSymmetric(dv_0);
      temp1.rightCols<3>() = gtsam::Matrix::Zero(3, 3);
      gtsam::Matrix36 temp2;
      temp2.leftCols<3>() = -Rw0 * gtsam::skewSymmetric(l0_);
      temp2.rightCols<3>() = Rw0;
      *H0 = gtsam::skewSymmetric(i_w - l_w) * temp1 + gtsam::skewSymmetric(dv_w) * temp2;
    }

    if (H1) {
      gtsam::Matrix36 temp;
      temp.leftCols<3>() = -Rw1 * gtsam::skewSymmetric(i1_);
      temp.rightCols<3>() = Rw1;
      *H1 = -gtsam::skewSymmetric(dv_w) * temp;
    }

    return residual;
  }
};

/**
 * @brief EdgeFactor with 3 keys so that j and l can come from different (not necessarily) frames
 *
 */
class GTSAM_EXPORT EdgeFactor3 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
{
  gtsam::Point3 j0_;  // In the frame of the first key
  gtsam::Point3 l1_;  // In the frame of the second key
  gtsam::Point3 i2_;  // In the frame of the third key

  bool same_keys_;

public:
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
  typedef EdgeFactor3 This;
  typedef boost::shared_ptr<This> shared_ptr;

  EdgeFactor3(
    const gtsam::Point3 & j0, gtsam::Key k0, const gtsam::Point3 & l1, gtsam::Key k1,
    const gtsam::Point3 & i2, gtsam::Key k2, const gtsam::SharedNoiseModel & model)
  : Base(model, k0, k1, k2), j0_(j0), l1_(l1), i2_(i2), same_keys_(k0 == k1)
  {
  }

  virtual ~EdgeFactor3() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & p0, const gtsam::Pose3 & p1, const gtsam::Pose3 & p2,
    boost::optional<gtsam::Matrix &> H0 = boost::none,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override
  {
    gtsam::Matrix36 Hi, Hj, Hl;
    bool compute_jacobians = H0 || H1 || H2;
    gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
    gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
    gtsam::Point3 i_w = p2.transformFrom(i2_, compute_jacobians ? &Hi : 0);
    gtsam::Vector3 dv_w = (j_w - l_w) / (j_w - l_w).norm();

    gtsam::Vector3 residual = (i_w - l_w).cross(dv_w);

    // All the jacobians not mentioned here are necessarily zero
    const gtsam::Matrix36 Hi_2 = Hi;
    const gtsam::Matrix36 Hj_0 = Hj;
    gtsam::Matrix36 Hj_1, Hl_0;
    if (same_keys_) {
      Hj_1 = Hj;
      Hl_0 = Hl;
    } else {
      Hj_1 = gtsam::Matrix::Zero(3, 6);
      Hl_0 = gtsam::Matrix::Zero(3, 6);
    }
    const gtsam::Matrix36 Hl_1 = Hl;

    if (H0) {
      gtsam::Matrix36 temp = gtsam::skewSymmetric(i_w - l_w) *
                               (gtsam::Matrix33::Identity() - dv_w * dv_w.transpose()) *
                               (Hj_0 - Hl_0) / (j_w - l_w).norm() +
                             gtsam::skewSymmetric(dv_w) * Hl_0;
      *H0 = temp;
    }

    if (H1) {
      gtsam::Matrix36 temp = gtsam::skewSymmetric(i_w - l_w) *
                               (gtsam::Matrix33::Identity() - dv_w * dv_w.transpose()) *
                               (Hj_1 - Hl_1) / (j_w - l_w).norm() +
                             gtsam::skewSymmetric(dv_w) * Hl_1;
      *H1 = temp;
    }

    if (H2) {
      gtsam::Matrix36 temp = -gtsam::skewSymmetric(dv_w) * (Hi_2);
      *H2 = temp;
    }
    return residual;
  }
};

/**
 * @brief EdgeFactor with 3 keys so that j and l can come from different (not necessarily) frames
 *
 */
class GTSAM_EXPORT EdgeFactor3S : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
{
  gtsam::Point3 j0_;  // In the frame of the first key
  gtsam::Point3 l1_;  // In the frame of the second key
  gtsam::Point3 i2_;  // In the frame of the third key

  bool same_keys_;

public:
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
  typedef EdgeFactor3S This;
  typedef boost::shared_ptr<This> shared_ptr;

  EdgeFactor3S(
    const gtsam::Point3 & j0, gtsam::Key k0, const gtsam::Point3 & l1, gtsam::Key k1,
    const gtsam::Point3 & i2, gtsam::Key k2, const gtsam::SharedNoiseModel & model)
  : Base(model, k0, k1, k2), j0_(j0), l1_(l1), i2_(i2), same_keys_(k0 == k1)
  {
  }

  virtual ~EdgeFactor3S() {}

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
      gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  gtsam::Vector evaluateError(
    const gtsam::Pose3 & p0, const gtsam::Pose3 & p1, const gtsam::Pose3 & p2,
    boost::optional<gtsam::Matrix &> H0 = boost::none,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override
  {
    gtsam::Matrix36 Hi, Hj, Hl;
    bool compute_jacobians = H0 || H1 || H2;
    gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
    gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
    gtsam::Point3 i_w = p2.transformFrom(i2_, compute_jacobians ? &Hi : 0);
    gtsam::Vector3 dv_w = (j_w - l_w) / (j_w - l_w).norm();

    gtsam::Vector3 e_v = (i_w - l_w).cross(dv_w);
    gtsam::Vector1 residual(e_v.norm());

    // All the jacobians not mentioned here are necessarily zero
    const gtsam::Matrix36 Hi_2 = Hi;
    const gtsam::Matrix36 Hj_0 = Hj;
    gtsam::Matrix36 Hj_1, Hl_0;
    if (same_keys_) {
      Hj_1 = Hj;
      Hl_0 = Hl;
    } else {
      Hj_1 = gtsam::Matrix::Zero(3, 6);
      Hl_0 = gtsam::Matrix::Zero(3, 6);
    }
    const gtsam::Matrix36 Hl_1 = Hl;

    if (H0) {
      gtsam::Matrix16 temp =
        (e_v.transpose() / e_v.norm()) *
        (gtsam::skewSymmetric(i_w - l_w) * (gtsam::Matrix33::Identity() - dv_w * dv_w.transpose()) *
           (Hj_0 - Hl_0) / (j_w - l_w).norm() +
         gtsam::skewSymmetric(dv_w) * Hl_0);
      *H0 = temp;
    }

    if (H1) {
      gtsam::Matrix16 temp =
        (e_v.transpose() / e_v.norm()) *
        (gtsam::skewSymmetric(i_w - l_w) * (gtsam::Matrix33::Identity() - dv_w * dv_w.transpose()) *
           (Hj_1 - Hl_1) / (j_w - l_w).norm() +
         gtsam::skewSymmetric(dv_w) * Hl_1);
      *H1 = temp;
    }

    if (H2) {
      gtsam::Matrix16 temp = (e_v.transpose() / e_v.norm()) * -gtsam::skewSymmetric(dv_w) * Hi_2;
      *H2 = temp;
    }
    return residual;
  }
};
}  // namespace gtsam
