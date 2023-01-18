#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam
{
  class GTSAM_EXPORT SurfaceFactor2
      : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
  {
    gtsam::Point3 j0_; // In the frame of the first key
    gtsam::Point3 l0_; // In the frame of the first key
    gtsam::Point3 m0_; // In the frame of the first key
    gtsam::Point3 i1_; // In the frame of the second key

  public:
    typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;
    typedef SurfaceFactor2 This;
    typedef boost::shared_ptr<This> shared_ptr;

    SurfaceFactor2(
        const gtsam::Point3 &j0, const gtsam::Point3 &l0, const gtsam::Point3 &m0, gtsam::Key k0, const gtsam::Point3 &i1, gtsam::Key k1,
        const gtsam::SharedNoiseModel &model)
        : Base(model, k0, k1),
          j0_(j0),
          l0_(l0),
          m0_(m0),
          i1_(i1)
    {
    }

    virtual ~SurfaceFactor2() {}

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p0, const gtsam::Pose3 &p1, boost::optional<gtsam::Matrix &> H0 = boost::none,
        boost::optional<gtsam::Matrix &> H1 = boost::none) const override
    {
      gtsam::Matrix36 Hi, Hj, Hl, Hm;
      bool compute_jacobians = H0 || H1;
      gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
      gtsam::Point3 l_w = p0.transformFrom(l0_, compute_jacobians ? &Hl : 0);
      gtsam::Point3 m_w = p0.transformFrom(m0_, compute_jacobians ? &Hm : 0);
      gtsam::Point3 i_w = p1.transformFrom(i1_, compute_jacobians ? &Hi : 0);
      gtsam::Vector3 n_w = (l_w - m_w).cross(j_w - m_w);
      gtsam::Vector3 n_w_hat = n_w / n_w.norm();
      gtsam::Vector3 residual = (i_w - m_w).dot(n_w_hat) * n_w_hat;

      if (H0)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj - Hm) -
                                     gtsam::skewSymmetric(j_w - m_w) * (Hl - Hm)) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm);

        *H0 = temp;
      }

      if (H1)
      {
        gtsam::Matrix36 temp = n_w_hat * n_w_hat.transpose() * Hi;

        *H1 = temp;
      }

      return residual;
    }
  };

  class GTSAM_EXPORT SurfaceFactor3JM
      : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
  {
    gtsam::Point3 j0_; // In the frame of the first key
    gtsam::Point3 l1_; // In the frame of the second key
    gtsam::Point3 m0_; // In the frame of the first key
    gtsam::Point3 i2_; // In the frame of the third key

  public:
    typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
    typedef SurfaceFactor3JM This;
    typedef boost::shared_ptr<This> shared_ptr;

    SurfaceFactor3JM(
        const gtsam::Point3 &j0, gtsam::Key k0, const gtsam::Point3 &l1, gtsam::Key k1,
        const gtsam::Point3 &m0, const gtsam::Point3 &i2, gtsam::Key k2,
        const gtsam::SharedNoiseModel &model)
        : Base(model, k0, k1, k2),
          j0_(j0),
          l1_(l1),
          m0_(m0),
          i2_(i2)
    {
    }

    virtual ~SurfaceFactor3JM() {}

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p0, const gtsam::Pose3 &p1, const gtsam::Pose3 &p2, boost::optional<gtsam::Matrix &> H0 = boost::none,
        boost::optional<gtsam::Matrix &> H1 = boost::none,
        boost::optional<gtsam::Matrix &> H2 = boost::none) const override
    {
      gtsam::Matrix36 Hi, Hj, Hl, Hm;
      bool compute_jacobians = H0 || H1 || H2;
      gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
      gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
      gtsam::Point3 m_w = p0.transformFrom(m0_, compute_jacobians ? &Hm : 0);
      gtsam::Point3 i_w = p2.transformFrom(i2_, compute_jacobians ? &Hi : 0);
      gtsam::Vector3 n_w = (l_w - m_w).cross(j_w - m_w);
      gtsam::Vector3 n_w_hat = n_w / n_w.norm();
      gtsam::Vector3 residual = (i_w - m_w).dot(n_w_hat) * n_w_hat;

      if (H0)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj - Hm) -
                                     gtsam::skewSymmetric(j_w - m_w) * (- Hm)) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm);

        *H0 = temp;
      }

      if (H1)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (-gtsam::skewSymmetric(j_w - m_w) * Hl) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat);

        *H1 = temp;
      }

      if (H2)
      {
        gtsam::Matrix36 temp = n_w_hat * n_w_hat.transpose() * Hi;

        *H2 = temp;
      }

      return residual;
    }
  };

  class GTSAM_EXPORT SurfaceFactor3LM
      : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
  {
    gtsam::Point3 j0_; // In the frame of the first key
    gtsam::Point3 l1_; // In the frame of the second key
    gtsam::Point3 m1_; // In the frame of the first key
    gtsam::Point3 i2_; // In the frame of the third key

  public:
    typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
    typedef SurfaceFactor3LM This;
    typedef boost::shared_ptr<This> shared_ptr;

    SurfaceFactor3LM(
        const gtsam::Point3 &j0, gtsam::Key k0, const gtsam::Point3 &l1, gtsam::Key k1,
        const gtsam::Point3 &m1, const gtsam::Point3 &i2, gtsam::Key k2,
        const gtsam::SharedNoiseModel &model)
        : Base(model, k0, k1, k2),
          j0_(j0),
          l1_(l1),
          m1_(m1),
          i2_(i2)
    {
    }

    virtual ~SurfaceFactor3LM() {}

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p0, const gtsam::Pose3 &p1, const gtsam::Pose3 &p2, boost::optional<gtsam::Matrix &> H0 = boost::none,
        boost::optional<gtsam::Matrix &> H1 = boost::none,
        boost::optional<gtsam::Matrix &> H2 = boost::none) const override
    {
      gtsam::Matrix36 Hi, Hj, Hl, Hm;
      bool compute_jacobians = H0 || H1 || H2;
      gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
      gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
      gtsam::Point3 m_w = p1.transformFrom(m1_, compute_jacobians ? &Hm : 0);
      gtsam::Point3 i_w = p2.transformFrom(i2_, compute_jacobians ? &Hi : 0);
      gtsam::Vector3 n_w = (l_w - m_w).cross(j_w - m_w);
      gtsam::Vector3 n_w_hat = n_w / n_w.norm();
      gtsam::Vector3 residual = (i_w - m_w).dot(n_w_hat) * n_w_hat;

      if (H0)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj)) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat);

        *H0 = temp;
      }

      if (H1)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (-Hm) -
                                     gtsam::skewSymmetric(j_w - m_w) * (Hl - Hm)) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm);

        *H1 = temp;
      }

      if (H2)
      {
        gtsam::Matrix36 temp = n_w_hat * n_w_hat.transpose() * Hi;

        *H2 = temp;
      }

      return residual;
    }
  };

  class GTSAM_EXPORT SurfaceFactor4
      : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
  {
    gtsam::Point3 j0_; // In the frame of the first key
    gtsam::Point3 l1_; // In the frame of the second key
    gtsam::Point3 m2_; // In the frame of the third key
    gtsam::Point3 i3_; // In the frame of the fourth key

  public:
    typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
    typedef SurfaceFactor4 This;
    typedef boost::shared_ptr<This> shared_ptr;

    SurfaceFactor4(
        const gtsam::Point3 &j0, gtsam::Key k0, const gtsam::Point3 &l1, gtsam::Key k1,
        const gtsam::Point3 &m2, gtsam::Key k2, const gtsam::Point3 &i3, gtsam::Key k3,
        const gtsam::SharedNoiseModel &model)
        : Base(model, k0, k1, k2, k3),
          j0_(j0),
          l1_(l1),
          m2_(m2),
          i3_(i3)
    {
    }

    virtual ~SurfaceFactor4() {}

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p0, const gtsam::Pose3 &p1, const gtsam::Pose3 &p2,
        const gtsam::Pose3 &p3, boost::optional<gtsam::Matrix &> H0 = boost::none,
        boost::optional<gtsam::Matrix &> H1 = boost::none,
        boost::optional<gtsam::Matrix &> H2 = boost::none,
        boost::optional<gtsam::Matrix &> H3 = boost::none) const override
    {
      gtsam::Matrix36 Hi, Hj, Hl, Hm;
      bool compute_jacobians = H0 || H1 || H2 || H3;
      gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
      gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
      gtsam::Point3 m_w = p2.transformFrom(m2_, compute_jacobians ? &Hm : 0);
      gtsam::Point3 i_w = p3.transformFrom(i3_, compute_jacobians ? &Hi : 0);
      gtsam::Vector3 n_w = (l_w - m_w).cross(j_w - m_w);
      gtsam::Vector3 n_w_hat = n_w / n_w.norm();
      gtsam::Vector3 residual = (i_w - m_w).dot(n_w_hat) * n_w_hat;

      if (H0)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * Hj) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat);

        *H0 = temp;
      }

      if (H1)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (-gtsam::skewSymmetric(j_w - m_w) * Hl) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat);

        *H1 = temp;
      }

      if (H2)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    ((gtsam::skewSymmetric(l_w - m_w) - gtsam::skewSymmetric(j_w - m_w)) * (-Hm)) /
                                    n_w.norm();

        gtsam::Matrix36 temp =
            ((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
            n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm);

        *H2 = temp;
      }

      if (H3)
      {
        gtsam::Matrix36 temp = n_w_hat * n_w_hat.transpose() * Hi;

        *H3 = temp;
      }

      return residual;
    }
  };

  class GTSAM_EXPORT SurfaceFactor4S
      : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
  {
    gtsam::Point3 j0_; // In the frame of the first key
    gtsam::Point3 l1_; // In the frame of the second key
    gtsam::Point3 m2_; // In the frame of the third key
    gtsam::Point3 i3_; // In the frame of the fourth key

    bool same_key_jl_;
    bool same_key_lm_;
    bool same_key_jm_;

  public:
    typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> Base;
    typedef SurfaceFactor4S This;
    typedef boost::shared_ptr<This> shared_ptr;

    SurfaceFactor4S(
        const gtsam::Point3 &j0, gtsam::Key k0, const gtsam::Point3 &l1, gtsam::Key k1,
        const gtsam::Point3 &m2, gtsam::Key k2, const gtsam::Point3 &i3, gtsam::Key k3,
        const gtsam::SharedNoiseModel &model)
        : Base(model, k0, k1, k2, k3),
          j0_(j0),
          l1_(l1),
          m2_(m2),
          i3_(i3),
          same_key_jl_(k0 == k1),
          same_key_lm_(k2 == k3),
          same_key_jm_(k0 == k3)
    {
    }

    virtual ~SurfaceFactor4S() {}

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override
    {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    gtsam::Vector evaluateError(
        const gtsam::Pose3 &p0, const gtsam::Pose3 &p1, const gtsam::Pose3 &p2,
        const gtsam::Pose3 &p3, boost::optional<gtsam::Matrix &> H0 = boost::none,
        boost::optional<gtsam::Matrix &> H1 = boost::none,
        boost::optional<gtsam::Matrix &> H2 = boost::none,
        boost::optional<gtsam::Matrix &> H3 = boost::none) const override
    {
      gtsam::Matrix36 Hi, Hj, Hl, Hm;
      bool compute_jacobians = H0 || H1 || H2 || H3;
      gtsam::Point3 j_w = p0.transformFrom(j0_, compute_jacobians ? &Hj : 0);
      gtsam::Point3 l_w = p1.transformFrom(l1_, compute_jacobians ? &Hl : 0);
      gtsam::Point3 m_w = p2.transformFrom(m2_, compute_jacobians ? &Hm : 0);
      gtsam::Point3 i_w = p3.transformFrom(i3_, compute_jacobians ? &Hi : 0);
      gtsam::Vector3 n_w = (l_w - m_w).cross(j_w - m_w);
      gtsam::Vector3 n_w_hat = n_w / n_w.norm();
      gtsam::Vector3 e_v = (i_w - m_w).dot(n_w_hat) * n_w_hat;
      gtsam::Vector1 residual(e_v.norm());

      const gtsam::Matrix36 Hi_3 = Hi;
      const gtsam::Matrix36 Hj_0 = Hj;
      gtsam::Matrix36 Hj_1, Hl_0;
      if (same_key_jl_)
      {
        Hj_1 = Hj;
        Hl_0 = Hl;
      }
      else
      {
        Hj_1 = gtsam::Matrix36::Zero();
        Hl_0 = gtsam::Matrix36::Zero();
      }
      gtsam::Matrix36 Hj_2, Hm_0;
      if (same_key_jm_)
      {
        Hj_2 = Hj;
        Hm_0 = Hm;
      }
      else
      {
        Hj_2 = gtsam::Matrix36::Zero();
        Hm_0 = gtsam::Matrix36::Zero();
      }
      const gtsam::Matrix36 Hl_1 = Hl;
      gtsam::Matrix36 Hl_2, Hm_1;
      if (same_key_lm_)
      {
        Hl_2 = Hl;
        Hm_1 = Hm;
      }
      else
      {
        Hl_2 = gtsam::Matrix36::Zero();
        Hm_1 = gtsam::Matrix36::Zero();
      }
      const gtsam::Matrix36 Hm_2 = Hm;

      if (H0)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj_0 - Hm_0) -
                                     gtsam::skewSymmetric(j_w - m_w) * (Hl_0 - Hm_0)) /
                                    n_w.norm();

        gtsam::Matrix16 temp =
            (e_v.transpose() / e_v.norm()) *
            (((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
             n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm_0));

        *H0 = temp;
      }

      if (H1)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj_1 - Hm_1) -
                                     gtsam::skewSymmetric(j_w - m_w) * (Hl_1 - Hm_1)) /
                                    n_w.norm();

        gtsam::Matrix16 temp =
            (e_v.transpose() / e_v.norm()) *
            (((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
             n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm_1));

        *H1 = temp;
      }

      if (H2)
      {
        gtsam::Matrix36 d_n_w_hat = (gtsam::Matrix33::Identity() - n_w_hat * n_w_hat.transpose()) *
                                    (gtsam::skewSymmetric(l_w - m_w) * (Hj_2 - Hm_2) -
                                     gtsam::skewSymmetric(j_w - m_w) * (Hl_2 - Hm_2)) /
                                    n_w.norm();

        gtsam::Matrix16 temp =
            (e_v.transpose() / e_v.norm()) *
            (((i_w - m_w).dot(n_w_hat)) * d_n_w_hat +
             n_w_hat * ((i_w - m_w).transpose() * d_n_w_hat - n_w_hat.transpose() * Hm_2));

        *H2 = temp;
      }

      if (H3)
      {
        gtsam::Matrix16 temp = (e_v.transpose() / e_v.norm()) * n_w_hat * n_w_hat.transpose() * Hi_3;

        *H3 = temp;
      }

      return residual;
    }
  };
} // namespace gtsam
