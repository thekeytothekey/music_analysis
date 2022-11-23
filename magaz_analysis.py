
import itertools
import pandas as pd
import re
import networkx as nx
from pyvis.network import Network
from IPython.display import display, HTML

def do_numbers(scale):
    """Converts scales in str format to list-of-bits format"""
    return [int(i) for i in scale]

def do_strings(scale):
    """Converts scales in list-of-bits format to str"""
    return ''.join([str(i) for i in scale])

def lxl(scale1, scale2):
    """Multiplies two lists of numbers"""
    if type(scale1) == str or type(scale2) == str:
        scale1, scale2 = do_numbers(scale1), do_numbers(scale2)
    return [int(a*b) for a,b in zip(scale1,scale2)]

def do_vector(scale):
    """Computes intervalic vector returning a str"""
    if type(scale) == str:
        scale = do_numbers(scale)
    vector = [sum(lxl(scale,scale[i+1:] + scale[:i+1])) for i in range(int(len(scale)/2))]
    weights = [1,1,1,1,1,0.5]
    return do_strings(lxl(vector, weights))

def do_profile(scale):
    """Computes intervalic profile returning a str"""
    if type(scale) == str:
        scale = do_numbers(scale)
    pos_notes = [i for i,n in enumerate(scale) if n==1]
    profile = [pos_notes[i]-pos_notes[i-1] for i in range(1, len(pos_notes))]
    profile.append((pos_notes[0]-pos_notes[-1])%12)
    return do_strings(profile)

def do_rootmode(scale):
    """Computes first-found rotation to transpose scale to C"""
    if type(scale) != str:
        scale = do_strings(scale)
    while scale[0]=="0":
        scale = scale[1:]+scale[0]
    return scale

def do_ring1(scale):
    """Computes all nearest neighbours scales (ring) from a given one"""
    if type(scale) == str:
        scale = do_numbers(scale)
    anl = []
    for i,n in enumerate(scale):
        if n:
            if not(scale[(i-1)%len(scale)]):
                anl.append([i,(i-1)%len(scale)])
            if not(scale[(i+1)%len(scale)]):
                anl.append([i,(i+1)%len(scale)])
    r = []
    for t in anl:
        scale2 = scale.copy()
        scale2[t[0]], scale2[t[1]] = scale[t[1]], scale[t[0]]
        r.append(do_rootmode(do_strings(scale2)))
    return {"scale":do_strings(scale),"analysis":anl, "ring":r}

def generate_data():
    """Generates a pandas dataframe with all scales of a given length"""
    l =12
    df = pd.DataFrame.from_dict({"scale":[''.join(j) for j in itertools.product(*["01"] * l)]})

    df = df[df.apply(lambda r: r.scale[0]!="0", axis=1)].reset_index(drop=True)

    df["length"] = df.apply(lambda r: sum(do_numbers(r.scale)), axis=1)
    df = df[df["length"]>1].reset_index(drop=True)

    df["profile"] = df.apply(lambda r: do_profile(r.scale), axis=1)
    df["vector"] = df.apply(lambda r: do_vector(r.scale), axis=1)

    df["dim_ring1"] = df.apply(lambda r: len(do_ring1(r.scale)["ring"]), axis=1)
    df["ring1"] = df.apply(lambda r: do_ring1(r.scale)["ring"], axis=1)
    return df.explode("ring1")

def df_merge_list(df, k, v):
    """Generates a pandas dataframe result of inner merging a pandas
    dataframe df with a list of values v on a key k"""
    return pd.merge(df,pd.DataFrame({k:v}),how="inner",on=k)

def df_find_ring1(df, s):
    """Generates a pandas dataframe result of filtering in a dataframe
    df with all scales inside ring of nearest neighbour scales"""
    return df_merge_list(df,"scale",do_ring1(s)["ring"])

def find_all(s,ss):
    """Find indices of all occurrences of substring ss in a string s"""
    return [m.start() for m in re.finditer(ss,s)]

def do_super_scales(scale):
    """Generates list of all super scales (with one note more) 
    from a given scale"""
    if type(scale) != str:
        scale = do_strings(scale)
    return [scale[:i]+"1"+scale[(i+1):] for i in find_all(scale,"0")]

def do_n_mode(scale,n):
    """Generates the n-th mode of the given scale"""
    if type(scale) != str:
        scale = do_strings(scale)
    i = find_all(scale,"1")[n]
    return scale[i:]+scale[:i]

def do_all_modes(scale):
    """Generates all modes of the given scale"""
    if type(scale) != str:
        scale = do_strings(scale)
    return [scale[i:]+scale[:i] for i in find_all(scale,"1")]

def generate_hepta_bas(c=True):
    """Generates all modes of basic heptatonics"""
    hpt_bas = {"Mn":"101011010101","MM":"101011011001",
                "ma":"101101011001","mm":"101101010101",
                "np":"110101010101","da":"110011011001"}
    hpt_bas_colors = {"MM":"black","Mn":"green",
                    "ma":"orange","mm":"pink",
                    "np":"purple","da":"cyan"}
    if c:
        return {k:{"modes":do_all_modes(v),"color":hpt_bas_colors[k]}
                for k,v in hpt_bas.items()}
    else:
        return {k:do_all_modes(v) for k,v in hpt_bas.items()}

def do_sub_scales(scale):
    """Generates list of all sub scales (with one note less) 
    from a given scale"""
    if type(scale) != str:
        scale = do_strings(scale)
    return [scale[:i]+"0"+scale[(i+1):] for i in find_all(scale,"1")]

def is_mode(s1,s2):
    """Finds out wheter scale s1 is a mode of scale s2"""
    if type(s1) != str or type(s2) != str:
        s1, s2 = do_strings(s1), do_strings(s2)
    return s1 in do_all_modes(s2)

def do_ring_graph(df):
    """Generates a networkx graph from a pandas dataframe df
    which contains scales and nearest neighbour scales"""
    return nx.from_pandas_edgelist(df,
            source="scale",
            target="ring1",
            edge_attr="dim_ring1")

def change_color(n, l, color="red"):
    """Changes color of all nodes from a pyvis.network.Network n 
    that appear in the list l"""
    for i,node in enumerate(n.nodes):
        if node["label"] in l:
            n.nodes[i]["color"] = "red"
    return n

def plot_graph_notebook(g, l = None, name = "graph", 
    save = True, show = True, colab=True, buttons=False):
    """Generates (and shows) a HTML file containing IPython interactive plot 
    showing networkx graph g in which all nodes in list l
    are coloured differently"""
    if colab:
        net = Network(notebook=True, cdn_resources="remote")
    else:
        net = Network(notebook=True)
    net.show_buttons(filter_=buttons)
    net.from_nx(g)
    if l: net = change_color(net, l)
    if save: net.save_graph("{}.html".format(name))
    #if show: net.show("{}.html".format(name))
    if show: display(HTML("{}.html".format(name)))