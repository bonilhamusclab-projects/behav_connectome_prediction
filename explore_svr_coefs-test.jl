using Base.Test
using MLBase
using PyCall

include("explore_svr_coefs.jl")


py_r2_score = begin
  @pyimport sklearn.metrics as metrics
  metrics.r2_score
end

y_true = rand(5)
y_pred = rand(5)

@test_approx_eq r2_score(y_true, y_pred) py_r2_score(y_true, y_pred)


n_samples = 10::Int64

macro test_cv_num_samples(cv)
  :(@test_approx_eq num_samples($cv) n_samples)
end

@test_cv_num_samples Kfold(n_samples, 5)
@test_cv_num_samples RandomSub(n_samples, round(Int64, .8 * n_samples), 4)
@test_cv_num_samples LOOCV(n_samples)
strata = [repmat([:a, :b, :c, :d], 2); [:a, :b]]::Vector{Symbol}
@test_cv_num_samples StratifiedRandomSub(strata, 5, 3)
@test_cv_num_samples StratifiedKfold(strata, 2)
