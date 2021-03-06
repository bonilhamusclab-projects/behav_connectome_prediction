using Base.Test
using MLBase
using PyCall

include("testUtils.jl")

@data_include("svrBase.jl")


py_r2_score = begin
  @pyimport sklearn.metrics as metrics
  metrics.r2_score
end

y_true = rand(5)
y_pred = rand(5)

@test_approx_eq r2Score(y_true, y_pred) py_r2_score(y_true, y_pred)
