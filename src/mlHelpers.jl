using DataFrames
using Gadfly
using Lazy
using MLBase

typealias Functions Vector{Function}

type Pipeline
  fits::Functions
  predicts::Functions
  score_fn::Function
  truths::AbstractVector
  param_state::Function
  param_state!::Function

  Pipeline(fits::Functions, predicts::Functions,
                    score_fn::Function, truths::AbstractVector,
                    param_state::Function, param_state!::Function) = new(
      fits, predicts, score_fn, truths, param_state, param_state!)
end


function Pipeline{T <: Any}(fits::Functions, predicts::Functions,
                  score_fn::Function, truths::AbstractVector,
                  model_state::Dict{Symbol, T})

  paramState!(p::ParamState) = model_state[p[1]] = p[2]

  paramState(s::Symbol) = model_state[s]

  Pipeline(fits, predicts, score_fn, truths, paramState, paramState!)
end


typealias IXs AbstractVector{Int64}

_runFns(p::Pipeline, f::Symbol, ixs::IXs, stop_fn::Int64) = foldl(
  ixs, p.(f)[1:stop_fn]) do prev_out, fn::Function
    fn(prev_out)
end

_runFns(p::Pipeline, f::Symbol, ixs::IXs) = _runFns(
  p, f, ixs, length(p.(f)))


function _runFns(p::Pipeline, f::Symbol, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64)
  fs = p.(f)
  first_out = fs[1](x_ixs, y_ixs=y_ixs)
  foldl( first_out, fs[2:stop_fn]) do prev_out, fn::Function
    fn(prev_out)
  end
end

_runFns(p::Pipeline, f::Symbol, x_ixs::IXs, y_ixs::IXs) = _runFns(
  p, f, x_ixs, y_ixs, p.(f) |> length)


typealias ParamState{T <: Any} Pair{Symbol, T}
typealias ModelState Dict{Symbol, Any}
paramState!(pipe::Pipeline, p::ParamState) = pipe.param_state!(p)
modelState!(pipe::Pipeline, m::ModelState) = map(p -> paramState!(pipe, p), m)

paramState(pipe::Pipeline, s::Symbol) = pipe.param_state(s)
modelState(pipe::Pipeline, ps::Vector{Symbol}) = ParamState[
                                                  p => paramState(pipe, p)
                                                  for p in ps]


pipeFit!(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :fits, ixs)
pipeFit!(pipe::Pipeline, ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :fits, ixs, stop_fn)

pipeFit!(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs) = _runFns(
  pipe, :fits, x_ixs, y_ixs)
pipeFit!(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :fits, x_ixs, y_ixs, stop_fn)



pipePredict(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :predicts, ixs)
pipePredict(pipe, ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :predicts, x_ixs, y_ixs, stop_fn)

pipePredict(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs) = _runFns(
  pipe, :predicts, x_ixs, y_ixs)
pipePredict(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs, stop_fn::Int64) = _runFns(
  pipe, :predicts, x_ixs, y_ixs, stop_fn)


function pipeTest(pipe::Pipeline, ixs::IXs)
  truths = pipe.truths[ixs]
  preds = pipePredict(pipe, ixs)
  pipe.score_fn(truths, preds)
end

function pipeTest(pipe::Pipeline, x_ixs::IXs, y_ixs::IXs)
  truths = pipe.truths[x_ixs]
  preds = pipePredict(pipe, x_ixs, y_ixs)
  pipe.score_fn(truths, preds)
end


function getTrainTestScores(pipe::Pipeline, cvg::CrossValGenerator,
    num_samples::Int64)

  num_iterations = length(cvg)

  train_scores = zeros(Float64, num_iterations)
  fit_call = 0

  function fit(ixs::IXs)
    fit_call += 1
    pipeFit!(pipe, ixs)
    train_scores[fit_call] = pipeTest(pipe, ixs)
  end

  test(_, ixs::IXs) = pipeTest(pipe, ixs)

  test_scores = cross_validate(fit, test, num_samples, cvg)
  train_scores, test_scores
end

typealias EvalInput{T <: AbstractVector} Pair{Symbol, T}
typealias Combos Vector{ModelState}
function _evalInputToModelStates(ei...)

  #enumerate hack to keep ordering
  need_splits::Vector{Int64} = @>> enumerate(ei) begin
    filter( ix_fv::Tuple{Int64, EvalInput} -> length(ix_fv[2][2]) > 1)
    map( ix_fv -> ix_fv[1])
  end

  if length(need_splits) == 0
    ms::ModelState = ModelState([k => v[1] for (k, v) in ei])
    Combos([ms])
  else
    ret = Combos()

    ix::Int64 = need_splits[1]
    f::Symbol, vs::AbstractVector = ei[ix]
    for v in vs
      remaining::Vector{EvalInput} = begin
        r = EvalInput[l for l in copy(ei)]
        r[ix] = f => [v]
        r
      end

      ret =[ret; _evalInputToModelStates(remaining...)]
    end

    ret
  end
end


function evalModelParallel(pipeGen::Function, cvg::CrossValGenerator, num_samples::Int64,
  states...)

  all_combos::Combos = _evalInputToModelStates(states...)

  scores = map(all_combos) do combo::ModelState
    pipe = pipeGen()
    modelState!(pipe, combo)
    train_scores, test_scores = getTrainTestScores(pipe, cvg, num_samples)
    mean(train_scores), mean(test_scores)
  end

  [t[1] for t in scores], [t[2] for t in scores], all_combos
end


function meanTrainTest{T <: AbstractVector}(train_test::Tuple{T, T})
  (mean(train_test[1]), mean(train_test[2]))
end


function evalModel(pipe::Pipeline, cvg::CrossValGenerator, num_samples::Int64,
    agg::Function=meanTrainTest, states...)

  all_combos::Combos = _evalInputToModelStates(states...)

  scores = map(all_combos) do combo::ModelState
    modelState!(pipe, combo)
    getTrainTestScores(pipe, cvg, num_samples) |> agg
  end

  Float64[t[1] for t in scores], Float64[t[2] for t in scores], all_combos
end


function plotEvalModel(train_scores, test_scores, labels)
  function mkLayer(scores, clr)
    color = eval(parse("colorant\"$clr\""))
    layer(y = scores, x = labels, Geom.point, Theme(default_color=color))
  end
  plot(mkLayer(train_scores, "deepskyblue"),
       mkLayer(test_scores, "green"),
       Guide.xlabel("Model State"),
       Guide.ylabel("Score"))
end

typealias Scores Vector{Float64}
function plotEvalModel{T}(modelEval::Tuple{Scores, Scores, Vector{T}})
  mkLabel(p::ParamState) = "$(p[1]): $(p[2])"
  mkLabel(m::ModelState) = join(map(mkLabel, m), "; ")
  labels = map(mkLabel, modelEval[3])
  plotEvalModel(modelEval[1], modelEval[2], labels)
end


sqDist(x, y) = norm( (y - x).^2, 1)

function r2score{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})

  dist_from_pred::Float64 = sqDist(y_true, y_pred)
  dist_from_mean::Float64 = sqDist(y_true, mean(y_true))

  1 - dist_from_pred/dist_from_mean
end


precisionScore(t_pos::Int64, f_pos::Int64) = t_pos/(t_pos + f_pos)

function precisionScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_pos = sum(y_pred & !y_true)

  precisionScore(true_pos, false_pos)
end

recallScore(t_pos::Int64, f_neg::Int64) = t_pos/(t_pos + f_neg)

function recallScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_neg = sum(!y_pred & y_true)

  recallScore(true_pos, false_neg)
end


function MLBase.f1score(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_true & y_pred)
  if true_pos == 0
    return 0
  end

  precision = precisionScore(y_true, y_pred)
  recall = recallScore(y_true, y_pred)

  2 * precision * recall / (precision + recall)
end


function MLBase.f1score{T}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})
  num_samples = length(y_true)
  labels = unique(y_true)

  reduce(0., labels) do acc::Float64, l::T
    truths = y_true .== l
    preds = y_pred .== l
    score = f1score(truths, preds)

    weight = sum(truths)/num_samples
    acc + score * weight
  end
end


calcCorrelations(dataf::DataFrame, predictors::Vector{Symbol},
  prediction::Symbol) = @>> predictors begin
    map(p -> cor(dataf[p], dataf[prediction]))
    cors -> DataFrame(predictor = predictors, cor = cors[:])
    sort(cols=[:cor])
  end


macro l2_from_true(x)
  :(sum( (y_true - $x).^2))
end


function r2Score{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})

  numerator::Float64 = @l2_from_true y_pred
  denominator::Float64 = @l2_from_true mean(y_true)

  1 - numerator/denominator
end
