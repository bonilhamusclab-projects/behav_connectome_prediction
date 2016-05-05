using DataFrames
using Gadfly


function plotComparison(compare_ret::DataFrame, 
                        yellow_col::Symbol, 
                        blue_col::Symbol;
                        predictor_col::Symbol = :lesion
                        )
    set_default_plot_size(10inch, 7inch)
    function mkLayer(col::Symbol, clr::ASCIIString)
        color = eval(parse("colorant\"$clr\""))
        layer(compare_ret, x=predictor_col, y=col, Theme(default_color=color), Geom.point)
    end
    
    plot(mkLayer(yellow_col, "yellow"), mkLayer(blue_col, "deepskyblue"))
end


function addRank!(df::DataFrame, col::Symbol, rank_name::Symbol=:rank)
    df = sort(df, cols=[col])
    df[rank_name] = collect(1:size(df, 1))
    df
end

addRank(df::DataFrame, col::Symbol, rank_name::Symbol=:rank) = addRank!(copy(df), col, rank_name)

immutable Param
    orig_t::Symbol
    new_t::Symbol
    orig_predictor::Symbol
    rank_name::Symbol
    input_df::DataFrame
end
    
function compare(
    lr_df::DataFrame, 
    svr_df::DataFrame;
    old_lr_predictor::Symbol=:edge,
    old_svr_predictor::Symbol=:lesion_name,
    new_predictor::Symbol=:lesion)
    
    lr_param = Param(:tscore, :lr_t, old_lr_predictor, :lr_rank, lr_df)
    svr_param = Param(:t, :svr_t, old_svr_predictor, :svr_rank, svr_df)
    
    lr::DataFrame, svr::DataFrame = map([lr_param, svr_param]) do p::Param
        ret = p.input_df[[p.orig_t, p.orig_predictor]]
        (p.orig_predictor != new_predictor) && rename!(
            ret, p.orig_predictor, new_predictor)
            
        (p.orig_t != p.new_t) && rename!(ret, p.orig_t, p.new_t)
        addRank!(ret, p.new_t, p.rank_name)
    end
    
    join(lr, svr, on=new_predictor)
    
end

compare_svr(lr_df::DataFrame, svr_df::DataFrame) = compare(
    lr_df, svr_df, 
    old_lr_predictor=:edge, old_svr_predictor=:conn, 
    new_predictor=:edge
)
