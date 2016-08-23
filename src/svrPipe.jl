using MLBase
using PyCall

@everywhere include("mlHelpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR
LinearSVC = svm.LinearSVC
SVC = svm.SVC


function normalizeData(X::AbstractArray,
  feature_mns::AbstractVector, feature_stds::AbstractVector)

  @assert length(feature_mns) == length(feature_stds) == size(X, 2)

  (X .- feature_mns')./feature_stds'
end


updateStateMeanStdGen(state::Dict) = Xy::XY -> begin
  X, _ = Xy

  state[:feature_means] = mean(X, 1)[:]
  state[:feature_stds] = std(X, 1)[:]
  state[:variant_features] = state[:feature_stds] .> 1e-6

  Xy
end


doSomethingGen(fn, run) = run ? Xy::XY -> fn(Xy) : identity


normalizeDataGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  X, y = Xy

  norm_X = normalizeData(X, state[:feature_means], state[:feature_stds])

  norm_X, y
end


selectVariantColsGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  X, y = Xy
  variant_features = state[:variant_features]

  X[:, variant_features], y
end


updateStateBiasGen(state::Dict) = Xy::XY -> begin
  X, y = Xy

  state[:cors] = cor(X, y)
  state[:bias] = if sum(state[:cors] .< 0) > sum(state[:cors] .> 0)
    -1 * maximum(y)
  else
    -1 * minimum(y)
  end

  X, y
end


offsetBiasGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  Xy[1], Xy[2] + state[:bias]
end


reverseBiasGen(state::Dict, run) = y::Vector -> run ? y - state[:bias] : y


svrPipeline(X::Matrix, y::Vector) = classifierPipeline(X, y)


function classifierPipeline{T}(X::Matrix, y::Vector{T};
  normalize::Bool=false,
  select_variant_cols::Bool=true,
  classifier::PyObject = LinearSVR(),
  score_fn::Function = r2score
  )

  state = Dict{Symbol, Any}(:classifier => classifier, :C => classifier[:C])

  selectData(ixs::IXs; y_ixs::IXs=ixs) = X[ixs, :], y[y_ixs]

  selectVariantCols = selectVariantColsGen(state, select_variant_cols)

  normalizeData = normalizeDataGen(state, normalize)

  fit_fns::Functions = begin
    updateStateMeanStd! = updateStateMeanStdGen(state)

    function classifierFit!(Xy::XY)
      classifier[:C] = state[:C]
      curr_X::Matrix = Xy[1]
      curr_y::Vector{T} = Xy[2]
      classifier[:fit](curr_X, curr_y)
    end

    [selectData, updateStateMeanStd!,
      normalizeData, selectVariantCols,
      classifierFit!]
  end

  predict_fns::Functions = begin
    onlyX(Xy::XY) = Xy[1]
    mockY(X::Matrix) = (X, T[])

    normalizeXdata(X::Matrix) = X |> mockY |> normalizeData |> onlyX

    selectVariantColsX(X::Matrix) = X |> mockY |> selectVariantCols |> onlyX

    classifierPredict(X::Matrix) = classifier[:predict](X)

    [selectData, onlyX,
      normalizeXdata, selectVariantColsX,
      classifierPredict
      ]
  end

  Pipeline(fit_fns, predict_fns, score_fn, y, state)
end


function trainContinuousTestCategorical(pipe, toCat = arr -> arr .>= 0;
  state_keys::Vector{Symbol} = [:C, :classifier],
  on_continuous_pred_calc::Function = (arr, ixs, p) -> ()
  )

  state = ModelState()

  fit_fns = begin
    function fn!(x_ixs::IXs; y_ixs::IXs=x_ixs)
      for kv in state
        paramState!(pipe, kv)
      end

      pipeFit!(pipe, x_ixs, y_ixs)

      for k in state_keys
        state[k] = paramState(pipe, k)
      end
    end
    [fn!]
  end

  truths_cat = pipe.truths |> toCat

  predict_fns = begin
    predict_call = 0
    function fn(x_ixs::IXs; y_ixs::IXs = x_ixs)
      continuous_preds = pipePredict(pipe, x_ixs, y_ixs)

      predict_call += 1
      on_continuous_pred_calc(continuous_preds, y_ixs, predict_call)

      continuous_preds |> toCat
    end
    [fn]
  end

  Pipeline(fit_fns, predict_fns, accuracy, truths_cat, state)
end


function ensemblePipe(pipe_lesion, pipe_conn, predict_ix)
  fit_fns = begin
    function fitBoth(ixs)
      pipeFit!(pipe_lesion, ixs)
      pipeFit!(pipe_conn, ixs)
    end

    [fitBoth]
  end


  function decisionPredict(p, ixs)
    pre_pred = pipePredict(p, ixs, predict_ix - 1)
    paramState(p, :classifier)[:predict_proba](pre_pred)
  end

  predict_fns = begin
    function predict(ixs)
      lesion_preds = decisionPredict(pipe_lesion, ixs)
      conn_preds = decisionPredict(pipe_conn, ixs)
      class_1 = (lesion_preds[:, 1] + conn_preds[:, 1])./2
      class_2 = (lesion_preds[:, 2] + conn_preds[:, 2])./2

      (class_2 .> class_1)[:]
    end

    [predict]
  end

  Pipeline(fit_fns, predict_fns, pipe_lesion.score_fn, pipe_lesion.truths, ModelState())
end
