import pandas as pd
import matplotlib.pyplot as plt


def create_plot(x,y,x_label="",y_label="",title="",legend_label="",fig_ax = (None,None)):
    if fig_ax == (None,None):
        fig_ax = plt.subplots()
    fig,ax = fig_ax
    ax.plot(x,y,label=legend_label)
    fig.set_size_inches(12,12)
    ax.grid(True)
    ax.set_facecolor("silver")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return fig, ax


df = pd.read_csv("comparison_PL_W50.csv", header=0,sep=";\t")        # read data from csv file


df.sort_values(by="edges",inplace=True)
dd = df[df["correct_sol_time"] > 0].copy() # If the correct solution isn't calculated, don't plot it
edge_time = create_plot(
    df["edges"],
    df["my_sol_time"],
    legend_label="Approximation algorithm",
    )
edge_time = create_plot(
    dd["edges"],
    dd["correct_sol_time"],
    x_label="Number of edges",
    y_label="Time (s)",
    title="Time comparison of Dinic's algorithm and an approximation algorithm (Edges)",
    legend_label="Dinic's algorithm",
    fig_ax=edge_time
)

df.sort_values(by="vertices",inplace=True)
vertices_time = create_plot(
    df["vertices"],
    df["my_sol_time"],
    legend_label="Approximation algorithm",
)
dd.sort_values(by="vertices",inplace=True)
vertices_time = create_plot(
    dd["vertices"],
    dd["correct_sol_time"],
    x_label = "Number of vertices",
    y_label = "Time (s)",
    title="Time comparison of Dinic's algorithm and an approximation algorithm (Vertices)",
    legend_label="Dinic's algorithm",
    fig_ax = vertices_time
)

df.sort_values(by="average_weight_of_edges",inplace=True)
weight_time = create_plot(
    df["average_weight_of_edges"],
    df["my_sol_time"],
    legend_label = "Approximation algorithm",
)
dd.sort_values(by="average_weight_of_edges",inplace=True)
weight_time = create_plot(
    dd["average_weight_of_edges"],
    dd["correct_sol_time"],
    x_label="Average weight of edge",
    y_label="Time (s)",
    title = "Time comparison of Dinic's algorithm and an approximation algorithm (Average weight of edge)",
    legend_label="Dinic's algorithm",
    fig_ax = weight_time
)
    
    
average_error = 100*(dd["my_sol"] - dd["correct_sol"]) / dd["correct_sol"]
print("Average error:", average_error.mean())
plt.show()
print(df)