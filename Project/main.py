"""
Correlation matrix heatmap
==========================
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.cluster import hierarchy
import math
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot

def format_exponent(ax, axis='y', y_horz_alignment='left'):
    # Change the ticklabel format to scientific format
    # ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 4))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        if y_horz_alignment == 'right':
            x_pos = 1
        y_pos = 1.0
        horizontalalignment = y_horz_alignment
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
        if y_horz_alignment == 'right':
            offset_text = r'$\mathregular{10^{%d}}$x' % expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment)
    return ax


def plot_proximity_heatmap(product_space_orig, proximity_matr):
    """
    Given the proximity matrix and the product space matrix produces the heatmap (simply based on sroting)
    @param proximity:
    @param product_space_orig:
    @return:
    """
    x = sorted(product_space_orig)
    y = sorted(product_space_orig)
    intensity = [0]*len(product_space_orig)
    i = 0
    for product in x:
        intensity[i] = [0]*len(product_space_orig)
        j = 0
        for product2 in y:
            if product in proximity_matr and product2 in proximity_matr[product]:
                intensity[i][j] = proximity_matr[product][product2]
            elif product2 in proximity_matr and product in proximity_matr[product2]:
                intensity[i][j] = proximity_matr[product2][product]
            elif product == product2:
                intensity[i][j] = 1
            else:
                pass
            j += 1
        i += 1

    intensity = np.array(intensity)
    f, ax = plt.subplots(figsize=(5, 4))
    sns.set_style("ticks", {'axes.edgecolor': '.0', 'axes.facecolor': 'black'})
    fd = sns.heatmap(intensity, xticklabels=False, yticklabels=False, cmap="RdYlBu_r", square=True)

    f.text(0.865, 0.5, r"Proximity $\phi$", ha='right', va='center', rotation='vertical', fontsize=13)
    f.tight_layout()
    plt.savefig('data/proximityheat.pdf')


def plot_proximity_heatmap2(product_space_orig, proximity_matr):
    """
    Given the proximity matrix and the product space matrix produces the heatmap (based on average linkage)
    @param proximity:
    @param product_space_orig:
    @return:
    """
    x = sorted(product_space_orig)
    y = sorted(product_space_orig)
    intensity = [0]*len(product_space_orig)
    i = 0
    for product in x:
        intensity[i] = [0]*len(product_space_orig)
        j = 0
        for product2 in y:
            if product in proximity_matr and product2 in proximity_matr[product]:
                intensity[i][j] = proximity_matr[product][product2]
            elif product2 in proximity_matr and product in proximity_matr[product2]:
                intensity[i][j] = proximity_matr[product2][product]
            elif product == product2:
                intensity[i][j] = 1
            else:
                pass
            j += 1
        i += 1

    intensity = np.array(intensity)
    f, ax = plt.subplots(figsize=(5, 4))
    sns.set_style("ticks", {'axes.edgecolor': '.0', 'axes.facecolor': 'black'})
    fd = sns.clustermap(intensity, xticklabels=False, yticklabels=False, cmap="RdYlBu_r",
                        square=True, method='average', row_cluster=True, col_cluster=True, linewidths=0)

    fd.cax.set_position([.15, .2, .03, .45])
    plt.gca().invert_yaxis()
    plt.gca().set_yticklabels( [item.get_text() for item in plt.gca().get_yticklabels()], rotation=180 )
    plt.text(0.86, 0.56, r"Proximity $\phi$", ha='right', va='center', rotation='vertical', fontsize=16)
    fd.savefig("data/proximityheat21.pdf")


def plot_proximity_density(proximity_values):
    """
    Given the proximity matrix, plts the density
    @param proximity_matr:
    @return:
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    f, ax = plt.subplots(figsize=(5, 4))
    values, base = np.histogram(proximity_values, bins=np.logspace(-2.0, 0.1, 20))
    #evaluate the cumulative
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], len(proximity_values)-cumulative, color='green', marker='.',
            markersize=10, linewidth=2)
    ax.set_xlim([0.01, 1])
    ax.set_ylim([900, 1000000])
    ax.grid(True)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Number of Links', fontsize=14)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.tight_layout()
    #format_exponent(ax, 'y')
    plt.savefig('data/proximitydensity.pdf')


def plot_proximity_prob_distr(proximity_values):
    """
    Given the proximity values plots the probability density
    @param proximity_values:
    @return:
    """
    proximity_values = np.array(proximity_values)
    mpl.rcParams.update(mpl.rcParamsDefault)
    f, ax = plt.subplots(figsize=(5, 4))
    density = gaussian_kde(proximity_values)
    density.set_bandwidth(bw_method='silverman')
    density.set_bandwidth(bw_method=density.factor / 5.)
    xs = np.logspace(-3, 0, 30)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    plt.plot(xs, density(xs), color='blue', marker='o', markersize=6, linewidth=2, mfc="white", mec="blue")
    ax.set_xlim([0.001, 1])
    #ax.set_ylim([0, 0.1])
    ax.grid(True)
    plt.xlabel('Link weight', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.gca().set_xscale("log")
    plt.tight_layout()
    plt.savefig('data/proximityprbdensity.pdf')


def plot_network_represenation(proximity_matr):
    """
    Given the proximity matrix draws the simple network representation
    @param proximity_values:
    @return:
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure(figsize=(5, 3))
    G = nx.Graph()
    for rowitem in proximity_matr:
        G.add_node(rowitem) if rowitem not in G.nodes() else None
        for columnitem in proximity_matr[rowitem]:
            if columnitem != rowitem:
                G.add_node(columnitem) if columnitem not in G.nodes() else None
                if (rowitem, columnitem) not in G.edges():
                    G.add_edge(rowitem, columnitem, weight=proximity_matr[rowitem][columnitem]) if proximity_matr[rowitem][columnitem] > 0.5 else None

    nodestoremove = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(nodestoremove)
    pos = nx.spring_layout(G, k=0.099) # positions for all nodes
    nx.draw(G, pos, node_color='r', node_size=9, edge_color='0.2', width=0.8, with_labels=False, linewidths=0.5)
    plt.axis('off')
    #plt.tight_layout()
    plt.savefig('data/network05.pdf')


def leamerclass(product, leamer, forlegend=False):
    """
    Given the product returns the corresponding coloring according to the Leamer classification
    @param product:
    @return:
    """
    colormap = {}
    colormap["Petroleum"] = (255/255.0,0,0)
    colormap["Raw Materials"] = (255/255.0,102/255.0,0)
    colormap["Forest Products"] = (255/255.0,163/255.0,102/255.0)
    colormap["Tropical Agriculture"] = (255/255.0,255/255.0,0)
    colormap["Animal Products"] = (255/255.0,255/255.0,179/255.0)
    colormap["Cereals, etc."] = (112/255.0,219/255.0,112/255.0)
    colormap["Labor Intensive"] = (46/255.0,184/255.0,46/255.0)
    colormap["Capital Intensive"] = (0,0,255/255.0)
    colormap["Machinery"] = (102/255.0,255/255.0,255/255.0)
    colormap["Chemical"] = (223/255.0,128/255.0,255/255.0)

    if forlegend:
        return colormap

    for classification in leamer:
        if product in leamer[classification]:
            return colormap[classification]
    return colormap["Petroleum"]


def mst(product_space_orig, proximity_matr, lemerclass):
    """
    Given the proximity amtrix plots the maximum spanning tree
    @param proximity_matr:
    @return:
    """
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3)
    mpl.rcParams.update(mpl.rcParamsDefault)

    cst_node_color = []
    cst_node_size = []
    cst_edge_color = []
    x = sorted(product_space_orig)
    y = sorted(product_space_orig)
    intensity = {}
    csrmatr = []
    for i in range(len(product_space_orig)):
        csrmatr.append([])
        for j in range(len(product_space_orig)):
            csrmatr[i].append(100)

    i = 0
    for product in x:
        j = 0
        if product not in intensity:
            intensity[product] = {}
        for product2 in y:
            if product2 not in intensity[product]:
                intensity[product][product2] = 0
            csrmatr[i][j] = 100
            if (product in proximity_matr and product2 in proximity_matr[product]):
                intensity[product][product2] = proximity_matr[product][product2]
                csrmatr[i][j] = -1*proximity_matr[product][product2]
            elif product2 in proximity_matr and product in proximity_matr[product2]:
                if product2 not in intensity:
                    intensity[product2] = {}
                intensity[product2][product] = proximity_matr[product2][product]
                csrmatr[j][i] = -1*proximity_matr[product2][product]
            j += 1
        i += 1

    X = csr_matrix(csrmatr)
    Tcsr = minimum_spanning_tree(X)
    maxspntree = Tcsr.toarray()

    G = nx.Graph()
    i = 0
    for rowitem in x:
        j = 0
        for colitem in y:
            if maxspntree[i][j] < 0:
                G.add_node(rowitem) if rowitem not in G.nodes() else None
                cst_node_color.append(leamerclass(rowitem, lemerclass))
                G.add_node(colitem) if colitem not in G.nodes() else None
                cst_node_color.append(leamerclass(colitem, lemerclass))
                if intensity[rowitem][colitem] < 0.4:
                    cst_edge_color.append('c')
                    theweight = intensity[rowitem][colitem]/60
                elif 0.4 <= intensity[rowitem][colitem] < 0.55:
                    cst_edge_color.append('y')
                    theweight = intensity[rowitem][colitem]/30
                elif 0.55 <= intensity[rowitem][colitem] < 0.65:
                    cst_edge_color.append('b')
                    theweight = intensity[rowitem][colitem]*30
                elif intensity[rowitem][colitem] >= 0.65:
                    cst_edge_color.append('r')
                    theweight = intensity[rowitem][colitem]*120
                G.add_edge(rowitem, colitem, weight=theweight)
            j += 1
        i += 1

    #nodestoremove = [node for node in G.nodes() if G.degree(node) == 0]
    #G.remove_nodes_from(nodestoremove)

    """
    for ritem in proximity_matr:
        for citem in proximity_matr[ritem]:
            if (ritem, citem) not in G.edges() and (citem, ritem) not in G.edges():
                if proximity_matr[ritem][citem] >= 0.65:
                    cst_edge_color.append('r')
                    theweight = intensity[ritem][citem]*20
                    G.add_edge(ritem, citem)
    """

    from networkx.drawing.nx_agraph import pygraphviz_layout
    #nx.drawing.nx_agraph.write_dot(G, "myfile")
    fontP = FontProperties()
    fontP.set_size('xx-small')
    pos = nx.spring_layout(G, iterations=800, scale=4500)
    ax1 = plt.subplot(gs[:-1, :])
    #pos = nx.spring_layout(G, k=0.009) # positions for all nodes
    nx.draw(G, pos, node_color=cst_node_color, node_size=9, edge_color=cst_edge_color, width=0.8, with_labels=False, linewidths=0.5)
    ax1.axis('off')
    plt.axis('off')
    patches = []
    ax2 = plt.subplot(gs[-1, :-1])
    patches.append(mpatches.Patch(color=(255/255.0,0,0), label='Petroleum'))
    patches.append(mpatches.Patch(color=(255/255.0,102/255.0,0), label='Raw Materials'))
    patches.append(mpatches.Patch(color=(255/255.0,163/255.0,102/255.0), label='Forest Products'))
    patches.append(mpatches.Patch(color=(255/255.0,255/255.0,0), label='Tropical Agriculture'))
    patches.append(mpatches.Patch(color=(255/255.0,255/255.0,179/255.0), label='Animal Products'))
    patches.append(mpatches.Patch(color=(112/255.0,219/255.0,112/255.0), label='Cereals, etc.'))
    patches.append(mpatches.Patch(color=(46/255.0,184/255.0,46/255.0), label='Labor Intensive'))
    patches.append(mpatches.Patch(color=(0,0,255/255.0), label='Capital Intensive'))
    patches.append(mpatches.Patch(color=(102/255.0,255/255.0,255/255.0), label='Machinery'))
    patches.append(mpatches.Patch(color=(223/255.0,128/255.0,255/255.0), label='Chemical'))
    ax2.legend(title="node color (Leamer Classification)", handles=patches, loc=1, ncol=4, prop = fontP, fancybox=True, shadow=False)
    ax2.axis('off')
    patches = []
    ax3 = plt.subplot(gs[-1, -1])
    patches.append(mpatches.Patch(color='red', label=r'$\phi > 0.65$'))
    patches.append(mpatches.Patch(color='blue', label=r'$\phi > 0.55$'))
    patches.append(mpatches.Patch(color='yellow', label=r'$\phi > 0.4$'))
    patches.append(mpatches.Patch(color='cyan', label=r'$\phi < 0.4$'))
    ax3.legend(title="link color (proximity)", handles=patches, loc=1, ncol=2, prop = fontP, fancybox=True, shadow=False)
    ax3.axis('off')
    #plt.tight_layout()
    plt.savefig('data/networkmst.pdf')


def plotMSTlegends():
    """
    plots the MST legends
    @return:
    """
    import matplotlib.gridspec as gridspec
    colors = ['r', 'b', 'y', 'c']
    G = nx.Graph()
    # some math labels
    labels = {}
    for i in range(5):
        G.add_node(i, pos=(i+1, 1))
        labels[i] = str(i)
    for i in range(1, 5):
        G.add_edge(i-1, i)
    pos = nx.get_node_attributes(G, 'pos')



    gs = gridspec.GridSpec(3, 3)
    #gs.update(left=0.05, right=0.48, wspace=0.05)

    ax3 = plt.subplot(gs[-1, -1])
    nx.draw(G, pos, linewidths=0.5, node_color='0.1', node_size=200, width=8, edge_color=colors, font_color='w')
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color='w')
    ax3.set_title('link color (proximity)')


    ax2 = plt.subplot(gs[-1, :-1])
    nodecolors = leamerclass("", "", forlegend=True)
    G = nx.Graph()
    # some math labels
    labels = {}
    for i in range(0, len(nodecolors)):
        G.add_node(i, pos=(i+1, 1))
        labels[i] = str(i+1)
    for i in range(1, len(nodecolors)):
        G.add_edge(i-1, i)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, linewidths=0.5, node_color=nodecolors.values(), node_size=300, width=2, edge_color='k', font_color='k')
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color='k')
    ax2.set_title('node color (Leamer Classification)')

    ax1 = plt.subplot(gs[:-1, :])
    nx.draw(G, pos)
    ax1.set_title('ff')
    plt.axis('off')
    #plt.tight_layout()
    plt.savefig('data/demo.pdf')


def read_database():
    """
    Reads the database into a pandas dataframe
    @param
    @return the dataframe
    """
    source = "data/wtf"
    mydata = pd.DataFrame()
    idxsum = 0
    year_range = ["98", "99", "00"]
    for year in year_range:
        temp = pd.read_csv(source + year + ".csv", header=0, sep='\t', error_bad_lines=False)
        mydata = pd.concat([mydata, temp])
        idxsum += len(temp)
    assert idxsum == len(mydata), "Houston we have got a problem, some data is lost out there in space"
    #mydata.sitc4 = mydata.sitc4.astype(np.str)
    product_space_orig = []
    country_space_orig = []
    rca_matrix_orig = {}
    proximity_matr = {}
    leamerclass = {}
    with open("data/proximity9800.txt") as f:
        for line in f:
            data = filter(len, line.split(' '))
            if data[0].replace("\"", "").rstrip('\n') not in proximity_matr:
                proximity_matr[data[0].replace("\"", "").rstrip('\n')] = {}
            proximity_matr[data[0].replace("\"", "").rstrip('\n')][data[1].replace("\"", "").rstrip('\n')] = float(data[2].rstrip('\n'))
            product_space_orig.append(data[0].replace("\"", "").rstrip('\n')) if data[0].replace("\"", "").rstrip('\n') not in product_space_orig else None
            product_space_orig.append(data[1].replace("\"", "").rstrip('\n')) if data[1].replace("\"", "").rstrip('\n') not in product_space_orig else None
    with open("data/leamer.txt") as f:
        for line in f:
            data = filter(len, line.split(' '))
            for i in range(len(data)):
                data[i] = data[i].replace("\"", "").rstrip('\n')
            if len(data) > 3:
                trtr = data[-1]
                del data[-1]
                data[-1] += " " + trtr
            if data[2] not in leamerclass and len(data[2]) > 2:
                leamerclass[data[2]] = []
            if len(data[2]) > 2:
                if data[0] not in leamerclass[data[2]]:
                    leamerclass[data[2]].append(data[0])
    with open("data/RCA.txt") as f:
        for line in f:
            data = filter(len, line.split(' '))
            country_space_orig.append(data[-1].replace("\"", "").rstrip('\n')) if data[-1].replace("\"", "").rstrip('\n') not in country_space_orig else None
            if data[-1].replace("\"", "").rstrip('\n') not in rca_matrix_orig:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')] = {}
            if data[0].replace("\"", "").rstrip('\n') not in rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')]:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')] = {}
            if int(data[2].replace("\"", "").rstrip('\n')) not in rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')] and 1998 <= int(data[2].replace("\"", "").rstrip('\n')) <= 2000:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')][int(data[2].replace("\"", "").rstrip('\n'))] = float(data[3].replace("\"", "").rstrip('\n'))
    return mydata, product_space_orig, country_space_orig, rca_matrix_orig, proximity_matr, leamerclass

numerical_year_range = range(1998, 2001)
mydata, product_space_orig, country_space_orig, rca_matrix_orig, proximity_matr, lemerclass = read_database()

#Here goes a small example of proximity calculation
product1 = "2114"
product2 = "0112"
numberij = 0
numberi = 0
numberj = 0
for country in rca_matrix_orig:
    product1sum = 0
    product2sum = 0
    for product in rca_matrix_orig[country]:
        if product == product1:
            product1sum = np.mean(rca_matrix_orig[country][product].values())
            if product1sum >= 1:
                numberi += 1
        if product == product2:
            product2sum = np.mean(rca_matrix_orig[country][product].values())
            if product2sum >= 1:
                numberj += 1
    if product1sum >= 1 and product2sum >= 1:
        numberij += 1

#proximityij = min{numberij/numberi, numberij/numberj}
#End of the example


product_space = []
for product in mydata.sitc4.unique():
    if len(str(product)) == 4 and not re.search('[a-zA-Z]', str(product)):
        product_space.append(str(product)) if str(product) not in product_space else None
    else:
        if '.' in str(product):
            t = str(product).rsplit('.', 1)[0]
            product_space.append(t) if t not in product_space else None
        else:
            if len(str(product)) > 3 and str(product)[0] in ['0', '1', '2', '3']:
                product_space.append(str(product)) if str(product) not in product_space else None

proximity_values = []
for key in proximity_matr:
    proximity_values.extend([item for item in proximity_matr[key].values()])

#plot_proximity_heatmap(product_space, proximity_matr)
#plot_proximity_heatmap2(product_space, proximity_matr)
#plot_proximity_heatmap2(product_space_orig, proximity_matr)
#plot_proximity_density(proximity_values)
#plot_proximity_prob_distr(proximity_values)
#plot_network_represenation(proximity_matr)
mst(product_space_orig, proximity_matr, lemerclass)
#plotMSTlegends()
exit()




