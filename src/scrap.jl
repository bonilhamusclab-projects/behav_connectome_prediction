using DataFrames
using Lazy
using UnicodePlots


include("DataInfo.jl")
include("helpers.jl")
include("mlHelpers.jl")
include("svrBase.jl")


picnicSeconds()=@> data_dir() joinpath("step2", "picnic_seconds.csv") readtable


function loadWps(ds::DataSet)
  di = DataInfo(adw, diff_wps, all_subjects, left_select2, ds)

  all_data = @> di getFull join(picnicSeconds(), on=:id)

  all_data[:pd_picnic_wps] = all_data[:pd_picnic_dw] ./ all_data[:picnic_seconds]

  all_data[:diff_wps] = all_data[:se_adw_wps] - all_data[:pd_picnic_wps]

  valid_rows = !all_data[:pd_picnic_wps].na

  all_data[valid_rows, :], getPredictors(di)
end


loadConn() = loadWps(conn)


loadLesion() = loadWps(lesion)


function loadAllData()
  conn_data, conn_preds = loadConn()
  lesion_data, lesion_preds = loadLesion()

  join(conn_data, lesion_data, on=:id), union(conn_preds, lesion_preds)
end


dropnan(arr) = arr[Bool[!isnan(i) for i in arr]]

histonan(arr) = @> arr dropnan histogram

plotCorrsGen(df, preds) = p -> calcCorrelations(df, preds, p)[:cor] |> histonan


function trainContinuousTestCategorical(pipe, toCat = arr -> arr .>= 0)
  fit_fns = Function[ixs -> pipeFit!(pipe, ixs)]

  truths_cat = pipe.truths |> toCat

  predict_fns = [ixs -> pipePredict(pipe, ixs) |> toCat]

  Pipeline(fit_fns, predict_fns, f1score, truths_cat, ModelState())
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
