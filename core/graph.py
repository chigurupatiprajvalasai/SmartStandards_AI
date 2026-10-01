import networkx as nx
import plotly.graph_objects as go

def standard_graph(df):
    G = nx.DiGraph()
    for _,r in df.iterrows():
        sid=r["standard_id"]
        G.add_node(sid, title=r["title"])
        for target in str(r["references"]).split(";"):
            target=target.strip()
            if target:
                G.add_edge(sid,target,relation="REFERENCES")
        for target in str(r["related_standards"]).split(";"):
            target=target.strip()
            if target:
                G.add_edge(sid,target,relation="RELATED")
        if str(r["supersedes"]).strip():
            G.add_edge(sid,str(r["supersedes"]).strip(),relation="SUPERSEDES")
    return G

def graph_figure(G, center):
    H = G.copy()
    nodes = {center}
    nodes |= set(H.predecessors(center)) if center in H else set()
    nodes |= set(H.successors(center)) if center in H else set()
    if len(nodes)==1:
        nodes = set(list(H.nodes)[:8])
    H=H.subgraph(nodes).copy()
    pos=nx.spring_layout(H, seed=42)
    edge_x=[]; edge_y=[]
    for u,v in H.edges():
        edge_x += [pos[u][0],pos[v][0],None]
        edge_y += [pos[u][1],pos[v][1],None]
    node_x=[pos[n][0] for n in H.nodes()]
    node_y=[pos[n][1] for n in H.nodes()]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=1),hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=list(H.nodes()),textposition="top center",
                             marker=dict(size=22),hovertext=[H.nodes[n].get("title","") for n in H.nodes()]))
    fig.update_layout(height=550, margin=dict(l=20,r=20,t=20,b=20), showlegend=False)
    return fig
