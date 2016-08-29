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
  pipe, :predicts, ixs, stop_fn)

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


@everywhere function trainTestPreds(pipe::Pipeline, cvg::CrossValGenerator)
  num_iterations = length(cvg)
  num_samples = length(pipe.truths)

  preds = zeros(Float64, num_samples)
  test_counts = zeros(Int64, num_samples)

  train_scores = zeros(Float64, num_iterations)
  fit_call = 0

  function fit(ixs::IXs)
    fit_call += 1
    pipeFit!(pipe, ixs)
    train_scores[fit_call] = pipeTest(pipe, ixs)
  end

  function test(_, ixs::IXs)
    test_counts[ixs] += 1
    preds[ixs] = (pipePredict(pipe, ixs) + (test_counts[ixs] - 1) .* preds[ixs])./test_counts[ixs]
    pipeTest(pipe, ixs)
  end

  test_scores = cross_validate(fit, test, num_samples, cvg)
  train_scores, test_scores, preds
end

typealias EvalInput{T <: AbstractVector} Pair{Symbol, T}
typealias Combos Vector{ModelState}
function stateCombos(ei...)

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

      ret =[ret; stateCombos(remaining...)]
    end

    ret
  end
end


meanTrainTest{T <: AbstractVector}(train::T, test::T) = mean(train), mean(test)


doNothing(train_scores::Vector, test_scores::Vector, preds::Vector, combo_ix::Int64) = ()


@everywhere function evalModel(pipe::Pipeline, cvg::CrossValGenerator,
    on_combo_complete::Function=doNothing,
    states...)
  state_combos::Combos = stateCombos(states...)
  evalModel(pipe, cvg, state_combos, on_combo_complete=on_combo_complete)
end


@everywhere function evalModel(pipe::Pipeline, cvg::CrossValGenerator,
    state_combos::Combos;
    on_combo_complete::Function=doNothing)

  scores::Vector{Tuple} = map(state_combos |> enumerate) do comboix_combo
    combo_ix::Int64, combo::ModelState = comboix_combo
    modelState!(pipe, combo)
    train_scores, test_scores, preds = trainTestPreds(pipe, cvg)

    on_combo_complete(train_scores, test_scores, preds, combo_ix)

    meanTrainTest(train_scores, test_scores)
  end

  train_scores = Float64[t[1] for t in scores]
  test_scores = Float64[t[2] for t in scores]

  train_scores, test_scores, state_combos
end


function evalModelParallel(X, y, pipe_factory::Function, cvg_factory::Function,
    state_combos::Combos;
    on_combo_complete::Function=doNothing)

  scores = pmap(state_combos) do c
    trains, tests, model = evalModel(pipe_factory(X, y), cvg_factory(y), [c])
    trains[1], tests[1], model[1]
  end

  train_scores = Float64[s[1] for s in scores]
  test_scores = Float64[s[2] for s in scores]
  combos = ModelState[s[3] for s in scores]

  train_scores, test_scores, combos
end



stringifyLabels(labels::Vector{ModelState}) = map(labels) do m::ModelState
  @> map(p::ParamState -> "$(p[1]): $(p[2])", m) join("; ")
end


function scoresLayer(scores, clr, x; include_smooth=true)
  color = "colorant\"$clr\"" |> parse |> eval
  geoms = include_smooth ? (Geom.point, Geom.smooth) : (Geom.point,)
  layer(y=scores, x=x, Theme(default_color=color), geoms...)
end


function plotPreds(truths, preds::Vector, subjects)
  perm_ixs = sortperm(truths)

  plot(scoresLayer(truths[perm_ixs], "deepskyblue", subjects[perm_ixs]),
   scoresLayer(preds[perm_ixs], "green", subjects[perm_ixs], include_smooth=false),
   Guide.xlabel("Subject"),
   Guide.ylabel("Score"))
end

function plotPreds(truths, model_eval, subjects)
  best_score_ix = model_eval[2] |> sortperm |> last
  preds = model_eval[4][:, best_score_ix]
  plotPreds(truths, preds, subjects)
end


function plotEvalModel(train_scores, test_scores, labels::Vector{ASCIIString})
  plot(scoresLayer(train_scores, "deepskyblue", labels),
       scoresLayer(test_scores, "green", labels),
       Guide.xlabel("Model State"),
       Guide.ylabel("Score"))
end


typealias Scores Vector{Float64}
function plotEvalModel(trains::Scores, tests::Scores, labels::Vector{ModelState})
  @>> labels stringifyLabels plotEvalModel(trains, tests)
end


plotEvalModel(model_eval) = plotEvalModel(model_eval[1], model_eval[2], model_eval[3])


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


function accuracy{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})
  @assert length(y_true) == length(y_pred)
  sum(y_true .== y_pred)/length(y_true)
end
