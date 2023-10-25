#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#LIBRARIES

# Basic Python Libraries
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import re

# Specific functions
from numpy.random import randint
import networkx.algorithms.community as cm

# Statistical Tools
from scipy.stats import t
from scipy.spatial.distance import squareform, pdist

# Factor Analysis Tools
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

# Clustering Algorithms
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.cluster import KMeans

# Clustering Performance Metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# Outlier Detection Algorithms
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class StructuralTypology:
    
    def __init__(self,dataset_path):
        

        # We initialize the color palette
        self.color, self.palette = self.define_color_palette()
        
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            INITIALIZATION OF THE DATASET
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        In the __init__ section, we translate the dataset to an adjacency matrix
        This part depends on the initial format of the data
        '''
        
        if dataset_path == False :
           
            # We define a dummy network just to get the names of the columns from 
            # the function to get all variables
            G = nx.Graph()
            G.add_nodes_from(np.arange(0,4,1))
            G.add_edge(0,1)
            columns_names = self.get_all_variables(G,0).columns
                
            # With this information, we define the Dataset              
            dataset = pd.DataFrame(columns=columns_names)
            
            # We import the names of the PNs in the folder
            PNs =  np.array(os.listdir('Data'))
            PNs = PNs[PNs!='.DS_Store']
            PNs = PNs[PNs!='dataset_with_outliers.csv']
            # Now we loop over the personal network of each node to extract the 
            # information    
            print('Computing the information about each Personal Network...')
            ids=np.array([])
            for PN in tqdm(PNs):
                # We keep the ID of the node
                id_node = int(re.findall(r'\d+', PN)[0])
                path = f'Data/{PN}'
                A = np.genfromtxt(path, delimiter=',')
    
                # We translate the Adjacency Matrix to NetworkX
                G = nx.from_numpy_array(A)
                if G.number_of_nodes()>2 and nx.density(G)>0:
                    # We obtain the variables we want
                    dataset = pd.concat([dataset,self.get_all_variables(G,id_node)])
                    ids = np.append(ids,id_node)
            
            self.ids = ids
            self.number_of_nodes=len(self.ids)
            self.nodes=np.linspace(0,self.number_of_nodes-1,self.number_of_nodes).astype(int)
                
            # WITH THIS ADJACENCY MATRIX, WE WANT TO CREATE THE DATASET WE WILL
            # WORK WITH. IT IS A DATASET IN WHICH EACH ROW REPRESENTS A NETWORK
            # AND EACH COLUMN REPRESENTS A VARIABLES OF THIS NETWORK  
            
            self.dataset = dataset.astype(float).copy()
            self.dataset_std = self.standardize_dataset(self.dataset).astype(float)
            
            self.dataset.to_csv('Data/dataset_with_outliers.csv')
    
            # We keep a copy of the complete dataset to try different outliers removal
            self.original_dataset = self.dataset.copy()
            self.original_dataset_std = self.dataset_std.copy()
        
        else:
            self.dataset = pd.read_csv('Data/dataset_with_outliers.csv',index_col = 'Unnamed: 0')
            self.dataset_std = self.standardize_dataset(self.dataset).astype(float)
             
            # We keep a copy of the complete dataset to try different outliers removal
            self.original_dataset = self.dataset.copy()
            self.original_dataset_std = self.dataset_std.copy()


    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # BASIC FUNCTIONS TO EXTRACT AND FORMAT THE DATA
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # FUNCTION TO STANDARDIZE THE DATASET 
    def standardize_dataset(self,dataset):
        # We compute the means for each column (feature)
        means=np.array(dataset.mean(axis=0))
        # We compute the standard deviations for each column (feature)
        stds=np.array(dataset.std(axis=0))
        # We save the number of rows
        n_row=dataset.shape[0]
        # We save the number of columns
        n_col=dataset.shape[1]
        # We create an array of ones with the length of the number of samples
        ones=np.ones(n_row)
        standarized_dataset=(dataset-np.transpose(means.reshape(n_col,1)*ones))/np.transpose(stds.reshape(n_col,1)*ones)
        return standarized_dataset
    
    # FUNCTION TO TRANSLATE THE ID TO THE CORRESPONDING NODE
    def id_to_node(self,ID):
        node=np.where(self.ids==ID)[0][0]
        return node

    # FUNCTION TO GET ALL THE VARIABLES GIVEN THE PERSONAL NETWORK OF A NODE
    def get_all_variables(self,G,id_node):

        '''
        BASIC VARIABLES
        '''
        # Size 
        n_nodes=float(G.number_of_nodes())
        # Density |||DEPENDENT ON N. NODES - POSSIBLE NORMALIZATION|||
        density=nx.density(G)
        '''
        COMPONENTS 
        '''
        # Number of isolated nodes |||DEPENDENT ON N. NODES - POSSIBLE NORMALIZATION|||
        n_isolated=float(nx.number_of_isolates(G))/n_nodes
        # Number of components |||DEPENDENT ON N. NODES - POSSIBLE NORMALIZATION|||
        n_components=float(nx.number_connected_components(G))
        '''
        TRIANGLES
        '''
        # WS clustering
        ws_clustering=nx.average_clustering(G)
        # Transitivity
        n_transitivity=nx.transitivity(G)
        # Share of triangles (the maximum number of triangles is (n over 3))
        max_triangles=n_nodes*(n_nodes-1)*(n_nodes-2)/6
        if max_triangles==0: n_triangles=0
        else: 
            absolute_n_triangles=sum(list(nx.triangles(G).values()))/3
            n_triangles=absolute_n_triangles/max_triangles


        # 1. Degree Centrality
        degree_centrality=np.array(pd.DataFrame.from_dict(nx.degree_centrality(G),orient='index')[0])
        median_dc=np.median(degree_centrality)
        # mean_dc=np.mean(degree_centrality) # It is the same as density when standardized
        std_dc=np.std(degree_centrality,ddof=1)
        centralization_dc=sum(max(degree_centrality)-degree_centrality)/((n_nodes-1)*(n_nodes-2))

        # 2. Closeness Centrality
        closeness_centrality=np.array(pd.DataFrame.from_dict(nx.closeness_centrality(G),orient='index')[0])
        median_cc=np.median(closeness_centrality)
        mean_cc=np.mean(closeness_centrality)
        std_cc=np.std(closeness_centrality,ddof=1)
        centralization_cc=sum(max(closeness_centrality)-closeness_centrality)/((n_nodes-1)*(n_nodes-2)/(2*n_nodes-3))


        # 3. Betweenness Centrality
        betweenness_centrality=np.array(pd.DataFrame.from_dict(nx.betweenness_centrality(G,normalized=True),orient='index')[0])
        median_bc=np.median(betweenness_centrality)
        mean_bc=np.mean(betweenness_centrality)
        std_bc=np.std(betweenness_centrality,ddof=1)
        centralization_bc=sum(max(betweenness_centrality)-betweenness_centrality)/((n_nodes-1))


        # 4. Eigenvector Centrality
        eigenvector_centrality=np.array(pd.DataFrame.from_dict(nx.eigenvector_centrality(G,max_iter=10000),orient='index')[0])
        median_ec=np.median(eigenvector_centrality)
        mean_ec=np.mean(eigenvector_centrality)
        std_ec=np.std(eigenvector_centrality,ddof=1)
        centralization_ec=sum(max(eigenvector_centrality)-eigenvector_centrality)/((n_nodes-1))

        # 5. PageRank Centrality
        pagerank=np.array(pd.DataFrame.from_dict(nx.pagerank(G,0.5),orient='index')[0])
        median_pc=np.median(pagerank)
        mean_pc=np.mean(pagerank)
        std_pc=np.std(pagerank,ddof=1)
        centralization_pc=sum(max(pagerank)-pagerank)/((n_nodes-1))

        # 6. Subgraph Centrality
        subgraph_centrality=np.array(pd.DataFrame.from_dict(nx.subgraph_centrality(G),orient='index')[0])
        median_sc=np.median(subgraph_centrality)
        mean_sc=np.mean(subgraph_centrality)
        std_sc=np.std(subgraph_centrality,ddof=1)
        centralization_sc=sum(max(subgraph_centrality)-subgraph_centrality)/((n_nodes-1))
        

        '''
        CONNECTIVITY
        '''
        n_bridges=len(list(nx.bridges(G)))

        global_efficiency=nx.global_efficiency(G)
        local_efficiency=nx.local_efficiency(G)

        '''
        ASSORTATIVITY
        '''
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        degree_assortativity=nx.degree_assortativity_coefficient(G)
        if degree_assortativity != degree_assortativity:
            degree_assortativity = 1
        
        warnings.simplefilter(action='always', category=RuntimeWarning)
            

        '''
        COMMUNITY DETECTION
        '''

        n_cliques=0
        for cl in nx.find_cliques(G):
            if len(cl)>3:
                n_cliques=n_cliques+1
                
        if density>0:
            # Girvan-Newman Communities
            max_modularity=0 
            optimal_partition=0
            k=0
            for partition in list(cm.centrality.girvan_newman(G)):
                modularity=cm.modularity(G,partition)
                if modularity>max_modularity: 
                    optimal_partition=k
                    max_modularity=modularity
                k=k+1
            girvan_newman_communities=list(cm.centrality.girvan_newman(G))[optimal_partition]
            modularity_gn=max_modularity
        
            n_com_1_gn=0
            n_com_2p_gn=0
            for community in girvan_newman_communities:
                if len(community)==1:
                    n_com_1_gn=n_com_1_gn+1
                else: 
                    n_com_2p_gn=n_com_2p_gn+1
                    
        
            # Louvain Communities
            louvain_communities=list(cm.louvain_communities(G))
            modularity_l=cm.modularity(G,louvain_communities)
            n_com_1_l=0
            n_com_2p_l=0
            for community in louvain_communities:
                if len(community)==1:
                    n_com_1_l=n_com_1_l+1
                else: 
                    n_com_2p_l=n_com_2p_l+1
            
        else:
            n_com_1_gn=n_nodes
            n_com_2p_gn=0
            modularity_gn=1
            n_com_1_l=n_nodes
            n_com_2p_l=0
            modularity_l=1
        
        n_com_1_gn = n_com_1_gn/n_nodes
        n_com_1_l = n_com_1_l/n_nodes
            

            
        '''
        STRUCTURAL HOLES
        '''
        constraint = np.array(pd.DataFrame.from_dict(nx.constraint(G),orient='index')[0].dropna())
        if len(constraint)==0: constraint=np.array([0,0]) #!!
        median_constraint=np.median(constraint)
        mean_constraint=np.mean(constraint)
        std_constraint=np.std(constraint,ddof=1)
        centralization_constraint=sum(max(constraint)-constraint)/(n_nodes-1)
        
        #effective_size=np.array(pd.DataFrame.from_dict(nx.effective_size(G),orient='index')[0].fillna(0))
        esize = nx.effective_size(G)
        efficiency = {n: v / G.degree(n) if G.degree(n)!=0 else 0 for n, v in esize.items()}
        effective_size=np.array(pd.DataFrame.from_dict(efficiency,orient='index')[0])
        median_es=np.median(effective_size)
        mean_es=np.mean(effective_size)
        std_es=np.std(effective_size,ddof=1)
        centralization_es=sum(max(effective_size)-effective_size)/(n_nodes-1)
        
        data = {'N. Nodes': n_nodes, #!!!!!!
                'Density': density,
                
                'N. Components': n_components,
                #'n_isolated': n_isolated, # We use as n_isolated the ones given by community partition algorithms
                
                'W-S Clustering': ws_clustering,
                'Transitivity': n_transitivity,
                'N. Triangles': n_triangles,
                
                #'median_dc' : median_dc, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\sigma$ (Degree Centrality)': std_dc,
                r'C* (Degree Centrality)': centralization_dc,

                #'median_cc' : median_cc, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\mu $(Closeness Centrality)': mean_cc,
                r'$\sigma$ (Closeness Centrality)': std_cc,
                r'C* (Closeness Centrality)': centralization_cc,
                
                # 'median_bc' : median_bc, # 0 Almost always #!!!
                r'$\mu$ (Betweenness Centrality)': mean_bc,
                r'$\sigma$ (Betweenness Centrality)': std_bc,
                r'C* (Betweenness Centrality)': centralization_bc,
                
                #'median_ec' : median_ec, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\mu$ (Eigenvector Centrality)': mean_ec,
                r'$\sigma$ (Eigenvector Centrality)': std_ec,
                r'C* (Eigenvector Centrality)': centralization_ec,
                
                #'median_pc' : median_pc, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\mu$ (PageRank)': mean_pc,
                r'$\sigma$ (PageRank)': std_pc,
                r'C* (PageRank)': centralization_pc,

                r'M (Subgraph Centrality)' : np.log(median_sc),
                r'$\mu$ (Subgraph Centrality)': np.log(std_sc),
                r'$\sigma$ (Subgraph Centrality)': np.log(mean_sc),
                r'C* (Subgraph Centrality)': np.log(centralization_sc),

                'N. Bridges': n_bridges,
                'Local Efficiency': local_efficiency,
                'Global Efficiency': global_efficiency,
                
                'Degree Assortativity': degree_assortativity,
                
                'N. Cliques': n_cliques,
                'Modularity (GN)': modularity_gn,
                'N. Isolated (GN)': n_com_1_gn, 
                'N. Communities 2+ (GN)': n_com_2p_gn,
                'Modularity (L)': modularity_l,
                'N. Isolated (L)': n_com_1_l, 
                'N. Communities 2+ (L)': n_com_2p_l,
                
                #'median_constraint' : median_constraint, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\mu$ (Constraint)': mean_constraint,
                r'$\sigma$ (Constraint)': std_constraint,
                r'C* (Constraint)': centralization_constraint,
                
                #'median_es' : median_es, # Symmetric Distribution, correlates with mean (density), multicollinearity #!!!
                r'$\mu$ (Effective Size)': mean_es,
                r'$\sigma$ (Effective Size)': std_es,
                r'C* (Effective Size)': centralization_es,
                }
          

        dataframe = pd.DataFrame(data,index=[id_node])

        return dataframe

    # COLOR PALETTE
    def define_color_palette(self):
        color = np.array([
        'royalblue',
        'darkorange',
        'forestgreen',
        'indianred',
        'mediumpurple',
        'gold',
        'cornflowerblue',
        'sandybrown',
        'limegreen',
        'red',
        'orchid',
        'goldenrod',
        'grey',
        'black'])
        
        palette = {
        -1: 'black',
        0: 'royalblue',
        1: 'darkorange',
        2: 'forestgreen',
        3: 'indianred',
        4: 'mediumpurple',
        5: 'gold',
        6: 'cornflowerblue',
        7: 'sandybrown',
        8: 'limegreen',
        9: 'red',
        10: 'orchid',
        11: 'goldenrod',
        12: 'grey',
        13: 'black',
        }
        return color, palette
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # EXPLORATORY DATA ANALYSIS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def depict_variables_distribution(self,y_limits = None):
        # We obtain an appropriate ordering of the variables
        linkage_data = linkage(self.dataset_std.T, method='ward', metric='euclidean')    
        D = dendrogram(linkage_data,labels=self.dataset_std.T.index)
        plt.close()
        features_order=D.get('ivl')
        dataset_violinplot=self.dataset_std[features_order].copy()

        plt.figure(figsize=(15,10))
        plt.grid(linewidth=0.5, alpha=0.5)
        # We depict the violins with the quartiles
        sns.violinplot(dataset_violinplot,
                       scale='width',
                       inner='quartile',
                       linewidth=1,
                       )
        # We depict the violins with the points
        sns.violinplot(dataset_violinplot,
                       scale='width',
                       inner='point',
                       linewidth=1,
                       )
    
        plt.xlabel('')
        plt.ylabel('Standarized values',fontsize=16)
    
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
    
        if y_limits is not None: plt.ylim(y_limits)
    
        plt.title('Variables PDFs', fontsize=12, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('Results/1_variables_distributions.pdf')
        plt.close()
    
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # CORRELATION ASSESSMENT
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def compute_and_plot_correlation_matrix(self):
        # We initialize the Figure
        plt.figure('Pearson Correlation Matrix')
    
        # We compute the correlation matrix
        corr_matrix = np.corrcoef(self.dataset_std,rowvar=False)
        # We save it in a DataFrame
        self.corr_mat = pd.DataFrame(corr_matrix,
                                     index = self.dataset_std.columns,
                                     columns = self.dataset_std.columns)
        # We plot a Heatmap of the correlation matrix
        hm = sns.heatmap(corr_matrix,
                         vmin=-1,vmax=+1, #Limit of the colorbar
                         annot=True,fmt='.2f',#We depict the value of the correlation coefficient
                         annot_kws={"fontsize":5},#We change the fontsize of the correlation coefficient
                         cmap='PRGn') 
    
        # We use as labels the names of the variables
        hm.set_xticks(np.arange(0,len(self.dataset_std.columns),1)+0.5)
        hm.set_yticks(np.arange(0,len(self.dataset_std.columns),1)+0.5)
        hm.set_xticklabels(self.dataset_std.columns)
        hm.set_yticklabels(self.dataset_std.columns)
        plt.xticks(fontsize=12,rotation=90)
        plt.yticks(fontsize=12,rotation=0)
    
        plt.title('PEARSON CORRELATION HEATMAP', fontsize=16, fontweight='bold')
    
        # We invert the orientation of the y axis
        #hm.invert_yaxis()
        # We modify the ticks of the colorbar
        cax = hm.figure.axes[-1]
        cax.tick_params(labelsize=20)
    
        # We display fullscreen the heatmap
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    
        plt.tight_layout()
        plt.show()
        plt.savefig('Results/1_correlation_heatmap.pdf')
        plt.close()
        

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # MACHINE LEARNING TOOLS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # OUTLIER DETECTION ALGORITHM
    def remove_outliers(self, hardness):
        
        dataset_std = self.original_dataset_std.copy()
        
        # We create the outliers dataframe 
        outliers=pd.DataFrame(columns=['IF','LOF','OCSVM'], index=dataset_std.index)

        # METHOD 1: ISOLATION FOREST
        IF = IsolationForest(max_samples= dataset_std.shape[0], 
                          contamination= 'auto',
                          max_features=len(dataset_std.columns)
                          )
        outliers_if = IF.fit_predict(dataset_std)
        # We save the result
        outliers['IF']=outliers_if
        
        # METHOD 2: LOCAL OUTLIER FACTOR
        LOF = LocalOutlierFactor(n_neighbors=100)
        outliers_lof = LOF.fit_predict(dataset_std)
        # We save the result
        outliers['LOF']=outliers_lof
        
        # METHOD 3: ONE-CLASS SVM        
        # We estimate the parameter nu as the average proportion of outliers found 
        # with the other two methods. 
        nu=((sum(outliers_if==-1)+sum(outliers_lof==-1))/2.)/dataset_std.shape[0] 
        OCSVM = OneClassSVM(nu=nu)
        outliers_ocsvm = OCSVM.fit_predict(dataset_std)
        # We save the result
        outliers['OCSVM']=outliers_ocsvm

        # Depending on the hardness specified, we can keep outliers in three 
        # different ways
        # Soft: we keep outliers if they appear in at least one of the three methods
        soft_outliers=outliers[outliers.sum(axis=1)<3].index 
        # Medium: we keep outliers if they appear in at least two of the three methods
        medium_outliers=outliers[outliers.sum(axis=1)<=-1].index
        # Hard: we keep outliers only if they appear in the three methods
        hard_outliers=outliers[outliers.sum(axis=1)==-3].index
        
        if hardness=='soft': outliers = soft_outliers
        elif hardness=='medium': outliers = medium_outliers
        elif hardness=='hard': outliers = hard_outliers

        print(f'We detect {len(outliers)} outliers')
        
        # We save the outliers ids
        self.outliers = outliers
        
        self.dataset = self.original_dataset.copy()
        # We separate these outliers from the ORIGINAL dataset
        self.outliers_dataset = self.dataset[self.dataset.index.isin(outliers)].copy()
        self.dataset = self.dataset[~self.dataset.index.isin(outliers)].copy()

        # We standarize again the reduced ORIGINAL dataset
        self.dataset_std = self.standardize_dataset(self.dataset)
        
        self.dataset.to_csv('Data/dataset.csv')
        self.dataset_std.to_csv('Data/dataset_std.csv')

    
    # VISUAL TEST TO CHECK OUTLIERS ELIMINATION HAS WORKED REASONABLY
    def assess_outliers_removal(self):  

        # First, we assign outliers a black color and inliers a yellow color 
        outliers_color=pd.DataFrame(columns=['outliers'], index=self.original_dataset.index)
        outliers_color.loc[self.outliers]=-1
        outliers_color.fillna(5,inplace=True)
    
        # We check, for instance, that outliers do not form a cluster and that the 
        # proportion of outliers is reasonable
        
        # Also, we check the spatial distribution of outliers
        
        # A PRINCIPAL COMPONENT PLOT
        # We perform a Singular Value Decomposition
        svd = np.linalg.svd(self.original_dataset_std,full_matrices=False)
        # We obtain U and D and V
        U=svd[0]
        D=np.diag(svd[1])
        V=np.transpose(svd[2])
        
        # We compute the cumulative variance explained by the components
        D2=np.matmul(D,D)
        percent_of_variance=np.diagonal(D2/D2.sum())
        cumulative_explained_variance=np.cumsum(percent_of_variance)
        n_components = sum(cumulative_explained_variance<0.9)

        # We are ploting in a PCA pairplot the location of outliers
        # Thus, we need the loadings to understand the dimensions in which outliers appear
        n = self.dataset_std.shape[0]
        load = np.matmul(V,np.sqrt(D2/(n-1)))
        # We save the loadings
        loadings_pca = pd.DataFrame(load, index=self.original_dataset.columns)
        # We keep the components we found with Parallel Analysis
        loadings_pca = loadings_pca[np.arange(n_components)]
        
        # We plot the pair-plot for n_components differentiating outliers
          
        #We compute the PCs
        PC=np.matmul(U,D)
        components = PC[:,0:n_components]
            
        data_pairplot=pd.DataFrame(components,index=self.original_dataset.index)
        data_pairplot['cluster']=outliers_color['outliers']
           
        pp = sns.pairplot(data_pairplot,
                          kind='scatter', 
                          diag_kind='kde', 
                          hue='cluster',
                          palette=self.palette,
                          plot_kws=dict(alpha=0.5)
                          )  
        pp.fig.suptitle("Outliers PCA Pair-plot")
        plt.tight_layout()
        plt.show()
        
        plt.savefig('Results/0_outliers_pca.pdf')
        plt.close()
        
        # We plot the loadings as well
        
        self.plot_loadings(loadings_pca,'Loadings of Outliers PCA')
        plt.savefig('Results/0_outliers_pca_loadings.pdf')
        plt.close()
    
        
    # KMeans Clustering
    def kmeans(self,n_clusters, dataset):
        
        k_means=KMeans(n_clusters=n_clusters,init='k-means++',n_init=10).fit(dataset)
        cluster_labels=k_means.labels_
        
        return cluster_labels


    def cluster_analysis(self):
        
        
        path_tail = '.pdf'

        dataset_to_cluster = self.factor_scores.copy()
        path_tail = f'_fa_{self.rotation}.pdf'
        

        # NOW WE STORE M REALIZATIONS FOR EACH ALGORITHM USING THE PROPER NUMBER 
        # OF VARIABLES IN EACH CASE, AND WE ANALYZE THE RESULTS
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # We define the cluster information dataframe
        cluster_information=pd.DataFrame(index=dataset_to_cluster.index)
        
        # KMEANS CLUSTERING 
        
        original_dataset_to_cluster = dataset_to_cluster.copy()
        
        # We define the number of clusters we will use for KMeans 
        clusters_kmeans = np.arange(2,15)      

        print('Silhouette Analysis for k-means')
            
        for n in clusters_kmeans:
            
            cluster_labels= self.kmeans(int(n), dataset_to_cluster)
            # The silhouette_score gives the average value for all the samples.
            # This gives insight into the density and separation of the formed
            # clusters. We will explore 'acceptable' values
            silhouette_avg = silhouette_score(dataset_to_cluster, cluster_labels, metric='euclidean')
            print(f"For n_clusters = {int(n)} the average silhouette score is {np.round(silhouette_avg,2)}")
        
        print(f'Clustering using KMeans in {2}-{15} clusters...')
        # Number of realizations we will store
        M=2
        for i in tqdm(range(M)):
            for n in clusters_kmeans:
                cluster_labels= self.kmeans(int(n), dataset_to_cluster)
                algorithm = str(int(n))+'means_'+str(i)
                cluster_information.loc[cluster_information.index,algorithm]=cluster_labels
                
                ordered_clusters = np.array(cluster_information[algorithm].value_counts().index)
                mapping = np.arange(len(ordered_clusters))
                cluster_information[algorithm].replace(ordered_clusters,mapping,inplace=True)

        dataset_to_cluster = original_dataset_to_cluster.copy()        
        cluster_information.to_csv('Results/cluster_information.csv')
        
        self.cluster_information = cluster_information
    
        # We need to keep a list of the names of the algorithms used
        algorithms = []
        for i in range(len(clusters_kmeans)):
            algorithms.append(str(int(clusters_kmeans[i]))+'means')
        
        M=50
        # We assess the performance computing the normalized mutual information between algorithms
        print('Computing the Normalized Mutual Information...')
        nmi=np.zeros((len(algorithms),len(algorithms)))    
        for i in tqdm(range(len(algorithms))): 
            for j in range(i,len(algorithms)): 
                algorithm_i=algorithms[i]
                algorithm_j=algorithms[j]
                
                nmi_ij=0
                counter=0
                for k in range(M):
                    feature_i=self.kmeans(int(algorithm_i[0:-5]), dataset_to_cluster)
                    feature_j=self.kmeans(int(algorithm_j[0:-5]), dataset_to_cluster)
                    nmi_ij=nmi_ij+normalized_mutual_info_score(feature_i,feature_j)
                    counter=counter+1
                nmi[i,j]=nmi_ij/counter
                nmi[j,i]=nmi_ij/counter
                        
        plt.figure('NMI')
        # We plot a Heatmap of the normalized mutual information
        hm = sns.heatmap(nmi,
                         vmin=0,vmax=+1, #Limit of the colorbar
                         annot=True,fmt='.2f',#We depict the value of the correlation coefficient
                         annot_kws={"fontsize":15},#We change the fontsize of the correlation coefficient
                         cmap='viridis') 
    
        # We use as labels the names of the variables
        hm.set_xticks(np.arange(0,len(algorithms),1)+0.5)
        hm.set_yticks(np.arange(0,len(algorithms),1)+0.5)
        hm.set_xticklabels(algorithms)
        hm.set_yticklabels(algorithms)
        plt.xticks(fontsize=12,rotation=90)
        plt.yticks(fontsize=12,rotation=0)
    
        plt.title('NMI among algorithms', fontsize=16, fontweight='bold')
    
        # We invert the orientation of the y axis
        hm.invert_yaxis()
        # We modify the ticks of the colorbar
        cax = hm.figure.axes[-1]
        cax.tick_params(labelsize=20)
    
        # We display fullscreen the heatmap
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    
        plt.tight_layout()
        plt.show()
        plt.savefig('Results/5_nmi'+path_tail)
        plt.close()
        
        self.path_tail = path_tail
        self.algorithms = algorithms


    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # FACTOR ANALYSIS AND PRINCIPAL COMPONENT ANALYSIS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # FUNCTION TO CHECK THE ASSUMPTIONS OF FACTOR ANALYSIS
    def check_factor_analysis_assumptions(self):

        # We perform the Bartlett's Sphericity Test to check if the correlation matrix 
        # is sufficiently different from an Identity Matrix
        chi_square_value,p_value = calculate_bartlett_sphericity(self.dataset_std)
        print(f"Bartlett's Sphericity (Chi2): {np.round(chi_square_value,1)} - p-value: {np.round(p_value,3)}")
        
        # We perform the KMO test to compare partial to total correlations, to make sure
        # that partial correlations are small with respect to total correlation
        kmo_all,kmo_model = calculate_kmo(self.dataset_std)
        kmo = pd.DataFrame()
        kmo['variable'] = self.dataset_std.columns 
        kmo['KMO'] = kmo_all
        kmo.set_index('variable',inplace=True)
        print('KMO for individual variables')
        print(kmo.sort_values(by='KMO'))
        print(f'Total KMO: {kmo_model}')

        
    # FUNCTION TO DETERMINE THE NUMBER OF FACTORS/COMPONENTS
    def perform_parallel_analysis(self):
        
        # Number of realizations
        K = 100
        # Factors extraction method
        method = 'principal'
        
        rotation = None
        # Rotation does not affect the final number of factors. It only rearranges the 
        # basis of the subspace, but the total explained variance by factor is constant
    
        # We extract the number of samples and the number of variables
        n, m = self.dataset_std.shape
        fa = FactorAnalyzer(n_factors = m, # For the moment we use the number of variables
                            rotation=rotation,method=method)
    
        # Eigenvalues arrays
        ev_list_fa = np.zeros((K,m)) 
        ev_list_pca = np.zeros((K,m))
        
        # Run the fit 'K' times over a random matrix (original matrix permutated)
        random_dataset = self.dataset_std.copy()
        print('Computing the number of Factors/Components')
        for i in tqdm(range(K)):
            # We randomly permutate the columns
            for column in random_dataset.columns:
                random_dataset[column]=np.random.permutation(random_dataset[column])
            # We fit the FA model
            fa.fit(random_dataset)
            # We fit the PCA model
            svd=np.linalg.svd(random_dataset,full_matrices=False)
            
            # We store the sorted list of eigenvalues
            ev_list_fa[i] = fa.get_eigenvalues()[0]
            ev_list_pca=svd[1]
            
        # We compute the 95%/50%/5% percentile for each ordinal eigenvalue 
        # (the largest, the second largest, etc.)
        percentiles_t_fa = np.zeros(m)
        percentiles_b_fa = np.zeros(m)
        percentiles_m_fa = np.zeros(m)
        for i in range(m):
            percentiles_t_fa[i] = np.percentile(ev_list_fa.T[i],95)
            percentiles_m_fa[i] = np.percentile(ev_list_fa.T[i],50)
            percentiles_b_fa[i] = np.percentile(ev_list_fa.T[i],5)
            
        percentiles_t_pca = np.zeros(m)
        percentiles_b_pca = np.zeros(m)
        percentiles_m_pca = np.zeros(m)
        for i in range(m):
            percentiles_t_pca[i] = np.percentile(ev_list_pca.T[i],95)
            percentiles_m_pca[i] = np.percentile(ev_list_pca.T[i],50)
            percentiles_b_pca[i] = np.percentile(ev_list_pca.T[i],5)
            
            
        # We fit the real dataset to the FA/PCA model as well
        fa.fit(self.dataset_std)
        svd=np.linalg.svd(self.dataset_std,full_matrices=False)
        # We get the real eigenvalues
        real_ev_fa = fa.get_eigenvalues()[0]
        real_ev_pca = svd[1]
    
        # We keep as relevant factors only those whose eigenvalues are above the 95% 
        # percentile of the expected random eigenvalue
        self.n_factors = np.sum(real_ev_fa > percentiles_t_fa)
        print(f'We find {self.n_factors} factors')
        self.n_components = np.sum(real_ev_pca > percentiles_t_pca)
        print(f'We find {self.n_components} components')
        
        # We plot the results
        x = np.arange(m)+1
        plt.figure('Parallel Analysis (FA)')
        plt.fill_between(x,percentiles_b_fa,percentiles_t_fa, color='royalblue',alpha=0.5, label='Randomized Matrix')
        plt.plot(x,percentiles_m_fa, color='royalblue')
        plt.plot(x,real_ev_fa, color = 'indianred',label = 'Real Eigenvalues')
        plt.axhline(1,  linestyle='--', color='black',label = 'Kaiser criterion')
        plt.xlabel('Number of factors',fontsize = 16)
        plt.ylabel('Eigenvalue',fontsize=16)
        plt.xticks(x,fontsize = 12)
        plt.yticks(fontsize=12)
        plt.yscale('log')
        plt.xlim([1,10])
        plt.ylim([0.9,3])
        plt.title(f'Parallel Analysis (Factor Analysis, rotation={rotation})',fontweight = 'bold')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig('Results/2_parallel_analysis_fa.pdf')
        plt.close()
    
        x = np.arange(m)+1
        plt.figure('Parallel Analysis (PCA)')
        D2t = percentiles_t_pca * percentiles_t_pca
        D2b = percentiles_b_pca * percentiles_b_pca
        plt.fill_between(x,D2b/D2b.sum(),D2t/D2t.sum(), color='royalblue',alpha=0.5, label='Randomized Matrix')
        D2 = percentiles_m_pca * percentiles_m_pca
        plt.plot(x,D2/D2.sum(), color='royalblue')
        D2 = real_ev_pca * real_ev_pca
        plt.plot(x,D2/D2.sum(),color = 'indianred', label = 'Real Eigenvalues')
        plt.xlabel('Number of components',fontsize = 16)
        plt.ylabel('Eigenvalue\n(Explained Variance)',fontsize=16)
        plt.xticks(x,fontsize = 12)
        plt.yticks(fontsize=12)
        plt.yscale('log')
        plt.xlim([1,10])
        plt.ylim([0.01,1])
        plt.title('Parallel Analysis (Principal Component Analysis)',fontweight = 'bold')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig('Results/2_parallel_analysis_pca.pdf')
        plt.close()
    
    def perform_principal_component_analysis(self):

        # We perform Singular Value Decomposition on the Dataset
        svd = np.linalg.svd(self.dataset_std,full_matrices=False)
        # We obtain U, V and D
        U=svd[0]
        V=np.transpose(svd[2])
        D=np.diag(svd[1])
    
        # We compute the cumulative variance explained by the components
        D2=np.matmul(D,D)
    
        # One thing we need to compute are the loadings matrix
        # Columns in matrix V are eigenvectors. This could be one option for a loading matrix 
        # However, we want loadings to represent correlation coefficients between variables and components
        # Thus, we need to save V * sqrt(explained variance)
        
        n = self.dataset_std.shape[0]
        load = np.matmul(V,np.sqrt(D2/(n-1)))
        # We save the loadings
        loadings_pca = pd.DataFrame(load, index=self.dataset_std.columns)
        # We keep the components we found with Parallel Analysis
        self.loadings_pca = loadings_pca[np.arange(self.n_components)]
    
        # Another useful thing is to express the data in the basis of Principal Axes
        # With this, for instance, we can do an interpretable pairplot, and color it 
        # when we do clustering 
        
        #We compute the PCs
        PC = np.matmul(U,D)
        self.components = PC[:,0:self.n_components]
        # We give names to the components
        names = []
        for i in range(self.n_components):
            names.append('PC'+str(i+1))
           
        pp = sns.pairplot(pd.DataFrame(self.components,columns=names),
                          kind='scatter', 
                          diag_kind='kde', 
                          #corner=True,
                          plot_kws=dict(color='darkslategrey',alpha=0.7),
                          diag_kws=dict(color='darkslategrey'),
                          height=20
                          )  
        pp.fig.suptitle("PCA Pair-plot")
        plt.tight_layout()
        plt.show()
        plt.savefig('Results/3_pairplot_pca.pdf')
        plt.close()
        
        self.plot_loadings(self.loadings_pca, 'PCA loadings')
        plt.savefig('Results/3_loadings_pca.pdf')
        plt.close()
    
    def perform_factor_analysis(self,rot=None):
        
        # Factor extraction method
        method = 'principal'
        # Rotations we will use
        rotations = [None, # 0 
                     'varimax', # 1 Orthogonal
                     'quartimax', # 2 Orthogonal
                     'equamax', # 3 Orthogonal
                     'oblimin' # 4 Oblique
                     ]
        
        rotation=rotations[rot]
        
        # Create factor analysis object and perform factor analysis
        fa = FactorAnalyzer(n_factors = self.n_factors, 
                            rotation=rotation,method=method).fit(self.dataset_std)
    
    
        self.loadings_fa=pd.DataFrame(fa.loadings_,index=self.dataset_std.columns)
    
        # The rotation matrix, if a rotation has been performed
        self.rotation_matrix = fa.rotation_matrix_
        # The factor correlations matrix. This only exists if rotation is ‘oblique’
        self.factor_correlation = fa.phi_
    
        # We compare Communalities with Uniquenesses
        self.variance_dataset = pd.DataFrame(index=self.dataset_std.columns)
        self.variance_dataset['communalities'] = fa.get_communalities()
        self.variance_dataset['uniquenesses'] = fa.get_uniquenesses()
        
        # We plot the result
        x = np.arange(self.dataset_std.shape[1])
        plt.figure('Variance')
        plt.plot(x,self.variance_dataset['communalities'],'o',color='royalblue',alpha=0.5,label='Communalities')
        plt.plot(x,self.variance_dataset['uniquenesses'],'o',color='indianred',alpha=0.5,label='Uniquenesses')
        plt.ylabel('Fraction of Correlation',fontsize=16)
        plt.xticks(x,self.dataset_std.columns,rotation = 90,fontsize=10)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'Results/4_variance_fa_{rotation}.pdf')
        plt.close()
        
        # We get the Eigenvalues
        # 1 Original Eigenvalues 
        # 2 Common Factor Eigenvalues
        original_eigenvalues, common_factor_eigenvalues = fa.get_eigenvalues()
        eigenvalues = pd.DataFrame()
        eigenvalues['original'] = original_eigenvalues
        eigenvalues['common_factors'] = common_factor_eigenvalues
    
        self.explained_variance = fa.get_factor_variance()
        # 1 Sum of squared loadings (variance)
        # 2 Proportional variance
        # 3 Cumulative variance
        print(f'Variance explained by the factors:\n{self.explained_variance[1]}')
    
        # We create names for the factors
        names = []
        for i in range(self.n_factors):
            names.append('Factor'+str(i+1))
            
        
        # Get factor scores for a new data set
        X = self.dataset_std.copy()
        self.factor_scores = pd.DataFrame(fa.transform(X),columns=names,index=X.index)
           
        pp = sns.pairplot(self.factor_scores,
                          kind='scatter', 
                          diag_kind='kde', 
                          #corner=True,
                          plot_kws=dict(color='darkslategrey',alpha=0.7),
                          diag_kws=dict(color='darkslategrey'),
                          height=20
                          )  
        pp.fig.suptitle(f"FA Pair-plot (rotation = {rotation})")
        plt.tight_layout()
        plt.show()
        plt.savefig(f'Results/4_pairplot_fa_{rotation}.pdf')
        plt.close()
        
        self.plot_loadings(self.loadings_fa, f'FA loadings (rotation = {rotation})')
        plt.savefig(f'Results/4_loadings_fa_{rotation}.pdf')
        plt.close()
        
        self.rotation = rotation
    

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # SPECIAL TYPES OF PLOTS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # CLUSTERED HEATMAP
    def outliers_clustered_heatmap(self, dataset):
        
        metric = 'euclidean'
        method = 'ward'
        
        # We obtain the list of ordered features using Hierarchical Clustering
        linkage_data = linkage(dataset.T, method=method, metric=metric)    
        D = dendrogram(linkage_data,labels=dataset.T.index)
        plt.close()        
        features_order=D.get('ivl')
        
        # We extract the ordered dataset
        dataset_clustermap=dataset[features_order].copy()
        
        # CLUSTERED HEATMAP   
        sns.clustermap(dataset_clustermap.T,
                       cmap="mako",

                      # Turn off the clustering
                      row_cluster=False, col_cluster=True,

                      # Make the plot look better when many rows/cols
                      linewidths=0, xticklabels=True, yticklabels=True, 
                      
                      # Ignore outliers
                      robust=True, 
                      # Remove colorbar
                      cbar_pos=None,
                      # Remove dendrograms
                      dendrogram_ratio=0.001)
        plt.yticks(rotation=0,fontsize=12)
        plt.tight_layout()
        
    # Function to plot the loadings    
    def plot_loadings(self,loadings,title):
        
        sns.clustermap(loadings,
                       cmap="BrBG",
                      
                      annot=True,fmt='.2f',#We depict the value of the correlation coefficient
                      annot_kws={"fontsize":5},#We change the fontsize of the correlation coefficient
    
                      # Make the plot look better when many rows/cols
                      linewidths=0, xticklabels=False, yticklabels=True, 
                      
                      cbar_pos=None,
                      dendrogram_ratio=0.001)
        plt.yticks(rotation=0,fontsize=12)
        plt.title(title,fontweight = 'bold')
        plt.tight_layout()
        
    def plot_clustering_results(self):
        
        for algorithm_short in tqdm(self.algorithms):
            algorithm = algorithm_short+'_'+str(randint(0,2))
            
        
            # Since we have indices going from 0, and cluster labels are also 0, 1, 2, etc.
            # We move temorarily the indices to be able to treat them together
            self.dataset.index = self.dataset.index + 20 #!!!
            self.dataset_std.index = self.dataset_std.index + 20 #!!!
            self.cluster_information.index = self.cluster_information.index + 20 #!!!
            
            # REORDERING OF THE CLUSTER LABELS
            
            # We want to represent variables in a way that similar variables appear together
            # To do so, we cluster them using Hirearchical Clustering 
            linkage_data = linkage(self.dataset_std.T, method='ward', metric='euclidean')    
            D = dendrogram(linkage_data,labels=self.dataset_std.T.index)
            plt.close()
            features_order=D.get('ivl')
            
            # We save the cluster labels ordered by size
            ordered_clusters = np.array(self.cluster_information[algorithm].value_counts().index)
            ordered_clusters = ordered_clusters[ordered_clusters!=-1]
            
            # We create a dataset with a column containing the information of the clustering
            clustered_dataset_std=pd.concat([pd.DataFrame(self.cluster_information[algorithm]),self.dataset_std[features_order]],axis=1)
            # We want all samples from the same clusters to appear together
            clustered_dataset_std=clustered_dataset_std.sort_values(by=algorithm).copy()
            
            # We do the same for the non std dataset
            clustered_dataset=pd.concat([pd.DataFrame(self.cluster_information[algorithm]),self.dataset[features_order]],axis=1)
            clustered_dataset=clustered_dataset.sort_values(by=algorithm).copy()
            
            # We assign the colors to the clusters
            col_colors_std=clustered_dataset_std[algorithm].map(self.palette)
            col_colors=clustered_dataset[algorithm].map(self.palette)
            
            confidence_level=0.95
            alpha_level=(1-confidence_level)/2
            # We compute the mean/std for each variable in each cluster    
            means_std=clustered_dataset_std.groupby(algorithm).mean()
            stds_std=clustered_dataset_std.groupby(algorithm).std(ddof=1)
            # We compute as well the square root of the number of samples in each cluster
            sqrt_M_std=clustered_dataset_std.groupby(algorithm).size().apply(np.sqrt)
            # With the information of the std and the nº of samples, we compute the error 
            # in the estimation of the mean for each variable in each cluster
            # We establish a confidence level of 95% and we compute the error        
            t_value_std=t.ppf(q=1-alpha_level,df=sqrt_M_std-1)
            prefactor=(t_value_std/np.array(sqrt_M_std))
            errors_std=stds_std.multiply(prefactor, axis=0)
            errors_std.fillna(0,inplace=True)
            
            # We do the same for the non-std dataset
            means=clustered_dataset.groupby(algorithm).mean()
            stds=clustered_dataset.groupby(algorithm).std(ddof=1)
            sqrt_M=clustered_dataset.groupby(algorithm).size().apply(np.sqrt)
            t_value=t.ppf(q=1-alpha_level,df=sqrt_M-1)
            prefactor=(t_value/np.array(sqrt_M))
            errors=stds.multiply(prefactor, axis=0)
            errors.fillna(0,inplace=True)
            
            # We need to save this dataset to plot the centroids
            dataset_with_means_std=pd.concat([self.dataset_std,means_std])
            dataset_with_means=pd.concat([self.dataset,means])
            
            # Finally, we drop the algorithm column
            clustered_dataset_std.drop(columns=[algorithm],inplace=True)
            clustered_dataset.drop(columns=[algorithm],inplace=True)
            
            # CLUSTERED HEATMAP   
            sns.clustermap(clustered_dataset_std.T,
                           cmap="mako",
        
                          # Turn off the clustering
                          row_cluster=False, col_cluster=False,
        
                          # Add colored class labels
                          col_colors=col_colors_std,
        
                          # Make the plot look better when many rows/cols
                          linewidths=0, xticklabels=False, yticklabels=True,
                          
                          robust=True,
                          cbar_pos=None,
                          dendrogram_ratio=0.001)
            plt.yticks(rotation=0,fontsize=12)
            plt.tight_layout()
            plt.savefig(f'Results/6_{algorithm}_clustermap.pdf')
            plt.close()
            
            # CENTROIDS COMPARISON      
            
            centroids_comparison_std = pd.DataFrame(columns = means_std.columns)
            k = 0
            c = 0
            # We choose cluster by descending size
            for cluster in pd.DataFrame(sqrt_M_std).sort_values(by=0,ascending=False).index: 
                centroids_comparison_std.loc[k] = means_std.loc[cluster] - errors_std.loc[cluster]
                centroids_comparison_std.loc[k+1] = means_std.loc[cluster]
                centroids_comparison_std.loc[k+2] = means_std.loc[cluster] + errors_std.loc[cluster]
                if cluster != -1:
                    centroids_comparison_std.loc[k,'cluster'] = c
                    centroids_comparison_std.loc[k+1,'cluster'] = c
                    centroids_comparison_std.loc[k+2,'cluster'] = c
                    c = c+1
                else:
                    centroids_comparison_std.loc[k,'cluster'] = -1
                    centroids_comparison_std.loc[k+1,'cluster'] = -1
                    centroids_comparison_std.loc[k+2,'cluster'] = -1
                
                k = k+3
                
            centroids_comparison = pd.DataFrame(columns = means.columns)
            k = 0
            c = 0
            # We choose cluster by descending size
            for cluster in pd.DataFrame(sqrt_M).sort_values(by=0,ascending=False).index: 
                centroids_comparison.loc[k] = means.loc[cluster] - errors.loc[cluster]
                centroids_comparison.loc[k+1] = means.loc[cluster]
                centroids_comparison.loc[k+2] = means.loc[cluster] + errors.loc[cluster]
                if cluster != -1:
                    centroids_comparison.loc[k,'cluster'] = c
                    centroids_comparison.loc[k+1,'cluster'] = c
                    centroids_comparison.loc[k+2,'cluster'] = c
                    c = c+1
                else:
                    centroids_comparison.loc[k,'cluster'] = -1
                    centroids_comparison.loc[k+1,'cluster'] = -1
                    centroids_comparison.loc[k+2,'cluster'] = -1
                
                k = k+3
                
            
            #new_title = 'cluster                                  '
            #centroids_comparison.rename(columns={'cluster':new_title},inplace=True)
            col_colors_std=centroids_comparison_std['cluster'].map(self.palette)
            centroids_comparison_std.drop(columns=['cluster'],inplace=True)
            col_colors=centroids_comparison['cluster'].map(self.palette)
            centroids_comparison.drop(columns=['cluster'],inplace=True)
        
            # CENTROIDS COMPARISON   
            sns.clustermap(centroids_comparison.T,
                           cmap="seismic_r",
        
                          # Turn off the clustering
                          row_cluster=False, col_cluster=False,
        
                          # Add colored class labels
                          col_colors=col_colors,
        
                          # Make the plot look better when many rows/cols
                          linewidths=0, xticklabels=False, yticklabels=True,
                          
                          vmin=-2.5,vmax=2.5,
                          robust=True,
                          cbar_pos=None,
                          annot = True,
                          annot_kws={"fontsize":5},
                          dendrogram_ratio=0.01
                          )
        
        
            plt.yticks(rotation=0,fontsize=12)
            plt.tight_layout()  
            plt.savefig(f'Results/6_{algorithm}_centroids.pdf',transparent=True)
            plt.close()
            
            # CENTROIDS COMPARISON   
            sns.clustermap(centroids_comparison_std.T,
                           cmap="seismic_r",
        
                          # Turn off the clustering
                          row_cluster=False, col_cluster=False,
        
                          # Add colored class labels
                          col_colors=col_colors_std,
        
                          # Make the plot look better when many rows/cols
                          linewidths=0, xticklabels=False, yticklabels=True,
                          
                          vmin=-2.5,vmax=2.5,
                          robust=True,
                          cbar_pos=None,
                          annot = True,
                          annot_kws={"fontsize":5},
                          dendrogram_ratio=0.01
                          )
        
        
            plt.yticks(rotation=0,fontsize=12)
            plt.tight_layout()  
            plt.savefig(f'Results/6_{algorithm}_centroids_std.pdf',transparent=True)
            plt.close()
            
            dataset_with_means_std.index = dataset_with_means_std.index.astype(int)
            distances=pd.DataFrame(squareform(pdist(dataset_with_means_std)), 
                         columns=dataset_with_means_std.index, 
                         index=dataset_with_means_std.index).loc[min(ordered_clusters):max(ordered_clusters),self.dataset_std.index]
            
            centroids = np.zeros(len(ordered_clusters))
            c = 0
            for i in distances.index:
                row=distances.loc[i]
                min_val=distances.loc[i][row==min(row)]
                centroids[c] = min_val.index[0]
                c=c+1
    
            c = min(ordered_clusters)
            for code in centroids:
                
                code = code-20 #!!!
                path = f'Datasets/PNs/{int(code)}.csv'
                A = np.genfromtxt(path, delimiter=',')
                G = nx.from_numpy_array(A)
                
                pos = nx.circular_layout(G)
                plt.figure(int(code))
                nx.draw_networkx(G,pos=pos,with_labels=False,node_size=40,node_color=self.palette[c],node_shape='h',width=1,edge_color='black')
                plt.axis("off")
                plt.savefig(f'Results/{self.palette[c]}.pdf',transparent = True)
                plt.close()
                c = c+1
            
            self.dataset.index = self.dataset.index - 20 #!!!
            self.dataset_std.index = self.dataset_std.index - 20 #!!!
            self.cluster_information.index = self.cluster_information.index - 20 #!!!
            
            ##############################################
            # PAIR-PLOT IN WHICH COLORS REPRESENT CLUSTERS
            ##############################################
            
            # We perform Singular Value Decomposition on the Dataset
            svd = np.linalg.svd(self.dataset_std,full_matrices=False)
            # We obtain U, V and D
            U=svd[0]
            D=np.diag(svd[1])
    
            # We use the number of components obtained in the PA    
            #We compute the PCs
            PC = np.matmul(U,D)
            components = PC[:,0:self.n_components]
            
            data_pairplot=pd.DataFrame(components,index=self.dataset_std.index)
            data_pairplot['cluster']=self.cluster_information[algorithm]
    
            pp = sns.pairplot(data_pairplot,
                              kind='scatter', 
                              diag_kind='kde', 
                              hue='cluster',
                              palette=self.palette,
                              #corner=True,
                              plot_kws=dict(alpha=0.5)
                              #plot_kws=dict(color='darkslategrey',alpha=0.7),
                              #diag_kws=dict(color='darkslategrey'),
                              #height=20
                              )  
            pp.fig.suptitle("PCA Pair-plot")
            plt.tight_layout()
            plt.savefig(f'Results/6_{algorithm}_clustered_pairplot.pdf',transparent=True)
            plt.close()

    


#%% 


if __name__ == '__main__':

    # USAGE

    # We load the class
    STclass = StructuralTypology(dataset_path = True)
        
    # We remove the outliers from the Dataset
    STclass.remove_outliers('medium')
    
    # We assess visually if the outliers removal has worked properly
    STclass.assess_outliers_removal()
    
    STclass.depict_variables_distribution()
    
    # We compute and plot the correlation matrix
    STclass.compute_and_plot_correlation_matrix()
    
    # We check the Factor Analysis assumptions
    STclass.check_factor_analysis_assumptions()
    
    # We perform a Parallel Analysis to determine the number of factors/components
    STclass.perform_parallel_analysis()
    
    # We perform Principal Component Analysis
    STclass.perform_principal_component_analysis()

    # NOW WE WILL PERFORM FACTOR ANALYSIS WITH 5 ROTATIONS, AND WE WILL DO CLUSTERING 
    # FOR EACH ONE OF THE ROTATIONS
    # ROTATIONS: 0 - None | 1 - Varimax | 2 - Quartimax | 3- Equamax | 4 - Oblimin
    
    # We perform Factor analysis 
    STclass.perform_factor_analysis(rot=4) # Specify rotation
    STclass.perform_factor_analysis(rot=1) # Specify rotation
    STclass.perform_factor_analysis(rot=0) # Specify rotation
    # We perform Clustering in the Factor's Subspace
    STclass.cluster_analysis()
    
    STclass.plot_clustering_results()
    
