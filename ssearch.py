from ntpath import join
import sys
from scipy import spatial
#Please, change the following path to where convnet2 can be located
#sys.path.append("..\convnet2")
import io
import tensorflow as tf
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np
import pandas as pd
import statistics
import umap
import umap.plot
import pathlib
import torch
import textsearch
from visual_text_parameters import parameters
from data_utils import prepare_dataset
from bpm_parameters import *
import sys
from PIL import Image, ImageDraw, ImageFont
from clip_ssearch import CLIPSSearch
from pprint import pprint

import networkx as nx
import matplotlib.pyplot as plt

import torch_geometric
import torch_geometric.data



#DATA_DIR = /home/vision/smb-datasets/VisualAttributes
#SEARCH_DIR = /home/vision/smb-datasets/VisualAttributes

class SSearch :
    def __init__(self, config_file, model_name):
        
        self.configuration = conf.ConfigurationFile(config_file, model_name)
        #defiing input_shape                    
        self.input_shape =  (self.configuration.get_image_height(), 
                             self.configuration.get_image_width(),
                             self.configuration.get_number_of_channels())                       
        #loading the model
        model = tf.keras.applications.ResNet50(include_top=True, 
                                               weights='imagenet', 
                                               input_tensor=None, 
                                               input_shape =self.input_shape, 
                                               pooling=None, 
                                               classes=1000)
        #redefining the model to get the hidden output
        #self.output_layer_name = 'conv4_block6_out'
        self.output_layer_name = 'avg_pool'
        output = model.get_layer(self.output_layer_name).output
        #output = tf.keras.layers.GlobalAveragePooling2D()(output)                
        self.sim_model = tf.keras.Model(model.input, output)        
        model.summary()
        #self.sim_model.summary()
        
        #defining image processing function
        #self.process_fun =  imgproc.process_image_visual_attribute
        self.process_fun =  imgproc.process_image
        #loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')        
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        self.enable_search = False        
        
    #read_image
    def read_image(self, filename):      
        #print("Reading {}".format(filename))  
        im = self.process_fun(data.read_image(filename, self.input_shape[2]), (self.input_shape[0], self.input_shape[1]))        
        #for resnet
        im = tf.keras.applications.resnet50.preprocess_input(im)    
        return im
    
    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')
        
    def load_catalog(self, catalog):
        with open(catalog) as f_in :
            data_path = os.path.abspath(self.configuration.get_data_dir())
            self.filenames = [os.path.join(data_path, "train", filename.strip()) for filename in f_in]
            # self.filenames = [filename.strip() for filename in f_in ]
        self.data_size = len(self.filenames)    
            
    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]
        
    def compute_features(self, image, expand_dims = False):
        #image = image - self.mean_image
        if expand_dims :
            image = tf.expand_dims(image, 0)        
        fv = self.sim_model.predict(image)            
        return fv
    
    def normalize(self, data) :
        """
        unit normalization
        """
        norm = np.sqrt(np.sum(np.square(data), axis = 1))
        norm = np.expand_dims(norm, 0)  
        #print(norm)      
        data = data / np.transpose(norm)
        return data
    
    def square_root_norm(self, data) :
        return self.normalize(np.sign(data)*np.sqrt(np.abs(data)))

    def adjust_query_embedding(self, query, original_embeddings, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "GlobalCategoryEN"].values[0]
                except:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "GlobalCategoryEN"].values[0]
                categories.append(category)
            
            adjust = all(x == categories[0] for x in categories)
            if adjust:
                #print("Decided to adjust")
                return new_query
            else:
                #print("Decided to NOT adjust")
                return query

        return new_query


    def adjust_query_embedding_sim(self, query, original_embeddings, text_model, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        #print(query.shape)
        #print("len(query[0]): ", len(query[0]))
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "ProductDescriptionEN"].values[0]
                except:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "ProductDescriptionEN"].values[0]
                #print("Description {}: {}".format(i, product_description))
                if text_model.model_name == "clip-base":
                    text_input = text_model.tokenizer(product_description, truncate=True).to(text_model.device)
                    with torch.no_grad():
                        text_features = text_model.model.encode_text(text_input)
                    text_features = text_features.cpu().numpy()[0]
                    data = text_features.astype(np.float32)
                    categories.append(data)
                elif text_model.model_name == "roberta":
                    data = text_model.model.encode(product_description)
                    categories.append(data)
                
            cos_sim_1 = 1 - spatial.distance.cosine(categories[0], categories[1])
            cos_sim_2 = 1 - spatial.distance.cosine(categories[0], categories[2])
            cos_sim_3 = 1 - spatial.distance.cosine(categories[1], categories[2])
            cos_sim_list = [cos_sim_1, cos_sim_2, cos_sim_3]
            adjust = all(cos_sim >= 0.8 for cos_sim in cos_sim_list)
            #adjust = True
            if adjust:
                #print("Decided to adjust")
                return new_query
            else:
                #print("NOT adjusting")
                return query
                #return new_query
        return new_query

 
    def search(self, im_query, metric = 'l2', norm = 'None', top=90, reducer=None, vtnn=None, adjust_query=False, adjust_query_sim=False, original_embeddings=None, df=None, text_model=None):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True)
        if adjust_query:
            q_fv = self.adjust_query_embedding(query=q_fv, original_embeddings=original_embeddings, top=3, df=df)
        if adjust_query_sim:
            q_fv = self.adjust_query_embedding_sim(query=q_fv, original_embeddings=original_embeddings, text_model=text_model, top=3, df=df)
        if vtnn is not None:
            q_fv = torch.tensor(q_fv)
            vtnn.eval()
            with torch.no_grad():
                q_fv = q_fv.to('cuda')
                q_fv = q_fv.view(-1, 2048)
                # q_fv = q_fv.view(-1, 1024)
                q_fv = vtnn(q_fv).cpu().numpy()
        #print("EMBEDDING SIZE: {}".format(len(q_fv[0])))
        #it seems that Euclidean performs better than cosine
        if metric == 'l2':
            if reducer is not None:
                data = reducer.transform(self.features)
                query = reducer.transform(q_fv)
            else:
                data = self.features
                query = q_fv
            if norm == 'square_root':
                data = self.square_root_norm(data)
                query = self.square_root_norm(query)
            print("Query features:", query.shape)
            try:
                print("data features:", data.shape)
            except:
                print("data features:", len(data), len(data[0]))
            d = np.sqrt(np.sum(np.square(data - query[0]), axis = 1))
            idx_sorted = np.argsort(d)
            d_sorted = np.sort(d)
        elif metric == 'cos' : 
            if norm == 'square_root':
                self.features = self.square_root_norm(self.features)
                q_fv = self.square_root_norm(q_fv)
            sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
            sim = np.reshape(sim, (-1))            
            idx_sorted = np.argsort(-sim)
            d_sorted = -np.sort(-sim)
            #print(sim[idx_sorted][:20])
        #print("idx_sorted: ", idx_sorted[:top])
        if top is not None:
            return idx_sorted[:top], d_sorted[:top], q_fv, data
        return idx_sorted, d_sorted, q_fv, data
                                
    def compute_features_from_catalog(self):
        n_batch = self.configuration.get_batch_size()        
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = np.float32)
        for i, filename in enumerate(self.filenames) :
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()
            images[i, ] = self.read_image(filename)        
        n_iter = np.int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter) :
            print('iter {} / {}'.format(i, n_iter))  
            sys.stdout.flush()             
            batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch))
        fvs = np.concatenate(result)    
        print('fvs {}'.format(fvs.shape))    
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames, write_data=False, similarity=None, distance=None):
        w = 1000
        h = 1000
        #w_i = np.int(w / 10)
        w_i = int(w / 10)
        #h_i = np.int(h / 10)
        h_i = int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            #y = np.int(np.floor(pos / w)) * h_i
            y = int(np.floor(pos / w)) * h_i
            image = data.read_image(filename, 3)
            
            if write_data:
                ### Add text with the product id
                try:
                    base = os.path.basename(filename)
                    filename = os.path.splitext(base)[0]
                    name_and_productid = filename.rsplit('_', 1)
                    font = ImageFont.truetype("arial.ttf", 30)
                    PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
                    draw = ImageDraw.Draw(PIL_image)
                    if (similarity is None and distance is None) or (i == 0):
                        draw.text((0, 0), "id: {}".format(name_and_productid[1]), font=font, fill='rgb(0, 0, 0)')
                    elif similarity is not None:
                        draw.text((0, 0), "id: {} / sim: {}".format(name_and_productid[1], round(similarity[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                    elif distance is not None:
                        draw.text((0, 0), "id: {} / dist: {}".format(name_and_productid[1], round(distance[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                except:
                    #print("Could not write id for product.")
                    pass
                image = np.array(PIL_image)

            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r    
                    
#unit test  

def get_ordered_relevants(r_filenames, dataframe, real_df=None):
    df = dataframe
    relevants = []

    base = os.path.basename(r_filenames[0])
    filename = os.path.splitext(base)[0]
    name_and_productid = filename.rsplit('_', 1)

    base_gc = real_df[real_df['Title'] == name_and_productid[0]]["GlobalCategoryEN"].values[0]
    dataframe['relevant'] = dataframe.apply(lambda x: 1 if base_gc == x['GlobalCategoryEN'] else 0, axis=1)
    for _, file in enumerate(r_filenames[1:]):
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        name_and_productid = filename.rsplit('_', 1)
        gc = df[(df['Title'] == name_and_productid[0])  & (df['ProductId'] == int(name_and_productid[1]))]['relevant'].values[0] # Pepeganga
        relevants.append(gc)
    return relevants


def get_product_and_category(r_filenames, dataframe, real_df=None):
    df = dataframe
    products = []
    for i, file in enumerate(r_filenames):
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        name_and_productid = filename.rsplit('_', 1)
        if real_df is not None:
            try:
                categories = df.loc[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1]), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            except:
                try: 
                    categories = df.loc[df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
                except:
                    categories = real_df.loc[real_df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            if i == 0:
                base_categories = categories
            else:
                file_info = [filename, categories[0], categories[1], categories[2]]
                products.append(file_info)

        else:
            try:
                categories = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            except: 
                try:
                    categories = df.loc[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1]), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
                except:
                    categories = df.loc[df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            if i == 0:
                base_categories = categories
            else:
                file_info = [filename, categories[0], categories[1], categories[2]]
                products.append(file_info)


    return base_categories, products


def recall_precision(relevants):
    n_relevant = 0
    recall = []
    precision = []
    n = len(relevants)
    pos = 1 
    for i, r in enumerate(relevants):
        if r:
            n_relevant += 1
            recall.append((i + 1)/n)
            precision.append(n_relevant/pos)
        pos += 1
    return recall, precision


def avg_precision(y, y_pred):
    p = 0
    n_relevant = 0
    pos = 1   
    p_tree = 0
    n_relevant_tree = 0
    pos_tree = 1
    p_sub = 0
    n_relevant_sub = 0
    pos_sub = 1
    for product in y_pred:
        if product[1] == y[0]:
            n_relevant += 1
            p += n_relevant / pos
        pos += 1
        if product[2] == y[1]:
            n_relevant_tree += 1
            p_tree += n_relevant_tree / pos_tree
        pos_tree += 1
        if product[3] == y[2]:
            n_relevant_sub += 1
            p_sub += n_relevant_sub / pos_sub
        pos_sub += 1
    
    if n_relevant != 0:
        ap = p / n_relevant
    else:
        ap = 0 

    if n_relevant_tree != 0:
        ap_tree = p_tree / n_relevant_tree
    else:
        ap_tree = 0

    if n_relevant_sub != 0:
        ap_sub = p_sub / n_relevant_sub
    else:
        ap_sub = 0 

    return ap, ap_tree, ap_sub

def remove_repeated(idx, dist_sorted):
    new_dist_sorted = []
    new_idx = []
    last_dist = None
    for i, d in enumerate(dist_sorted):
        if last_dist != d:
            new_dist_sorted.append(d)
            new_idx.append(idx[i])
            last_dist = d
    return new_idx, new_dist_sorted

def remove_repeated_files(r_filenames):
    new_filenames = []
    repeated_titles = []
    last_title = ""
    for path in r_filenames:
        file = os.path.basename(path).split(".")[0]
        title = file.split("_")[0]
        if last_title == title:
            repeated_titles.append(title)
            last_title = title
        if title not in repeated_titles:
            new_filenames.append(path)
    return new_filenames

def contrastive_loss(logits, adj, tau=1):
    similarity = 1 - tf.matmul(logits, logits, transpose_b=True)
    similarity = similarity / tau
    similarity = tf.exp(similarity)

    neighbours_mask = tf.cast(adj, dtype=tf.bool)

    non_neighb_simils = tf.where(neighbours_mask, tf.zeros_like(similarity), similarity)
    sum_non_neighb_simils = tf.reduce_sum(non_neighb_simils, axis=0)

    contrast = tf.boolean_mask(similarity, neighbours_mask)
    contrast = contrast / sum_non_neighb_simils[:, tf.newaxis]
    contrast = -tf.math.log(contrast)

    mean_contrast = tf.reduce_mean(contrast, axis=0)
    return mean_contrast

def gnn(fts, adj, transform, activation):
    seq_fts = transform(fts)
    # print("seq_fts shape", seq_fts.shape)

    ret_fts = tf.matmul(adj, seq_fts)
    # print("ret_fts shape", ret_fts.shape)

    return activation(ret_fts)

def train_cora(fts, adj, gnn_fn, units, epochs, lr):

    if units == None:
        units = fts.shape[1]

    lyr = tf.keras.layers.Dense(units)
    # lyr_2 = tf.keras.layers.Dense(7)

    # def cora_gnn(fts, adj):
    #     hidden = gnn_fn(fts, adj, lyr_1, tf.nn.relu)
    #     # logits = gnn_fn(hidden, adj, lyr_2, tf.identity)
    #     return hidden

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for op in range(epochs + 1):
        with tf.GradientTape() as t:
            logits = gnn(fts, adj, lyr, tf.nn.relu)
            # print("logits", logits.shape)
            loss = contrastive_loss(logits, adj)
            # print("loss", loss.shape)

        variables = t.watched_variables()
        # print("variables", variables.shape)

        grads = t.gradient(loss, variables)
        # print("grads", grads.shape)

        optimizer.apply_gradients(zip(grads, variables))

    return logits


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)                
    parser.add_argument("-mode", type=str, choices=['search', 'compute', 'eval-umap', 'eval-umap-clip', 'search-umap', 'eval-text', 'search-text-visual', 'eval-text-visual', 'eval-all-text-visual', 'eval-all-umap', 'utils', "eval-text-visual-clip", "gnn"], help=" mode of operation", required=True)
    parser.add_argument('-umap', action='store_true')
    parser.add_argument('-real', action='store_true', help="whether to use real images or not when evaluating")
    parser.add_argument("-dataset",  type=str, choices=['Pepeganga', 'PepegangaCLIPBASE', 'Cartier', 'CartierCLIPBASE', 'IKEA', 'IKEACLIPBASE', 'UNIQLO', 'UNIQLOCLIPBASE', 'WorldMarket', 'WorldMarketCLIPBASE', 'Homy', 'HomyCLIPBASE'], help="dataset", required=True)
    parser.add_argument("-list", type=str,  help=" list of image to process", required=False)
    parser.add_argument("-odir", type=str,  help=" output dir", required=False, default='.')
    pargs = parser.parse_args()     
    configuration_file = pargs.config        
    ssearch = SSearch(pargs.config, pargs.name)
    metric = 'l2'
    norm = 'None'
    
    dataset = pargs.dataset
    use_real_queries = pargs.real

    if pargs.mode == 'compute' :        
        ssearch.compute_features_from_catalog()
    
    if pargs.mode == 'graph':
        ssearch.load_features()
        n_components = 128
        text_data = np.load("./pytorch_labels/{}/UMAP/{}-dim/text_embeddings.npy".format(dataset, n_components))
        k = 6

        edge_idx_arr_0 = np.array([])
        edge_idx_arr_1 = np.array([])
        edge_feature_arr = np.array([])

        for product_idx in range(text_data.shape[0]):
            distances_by_axis = (text_data[product_idx,] - text_data) ** 2
            distance_to_product = np.sqrt(distances_by_axis.sum(axis=1))
            k_nearest_idxs = distance_to_product.argsort()[1:k+1]
            k_nearest_distances = distance_to_product[k_nearest_idxs,]

            edge_idx_arr_0 = np.append(edge_idx_arr_0, np.full(k, product_idx))
            edge_idx_arr_1 = np.append(edge_idx_arr_1, k_nearest_idxs)
            edge_feature_arr = np.append(edge_feature_arr, k_nearest_distances)

        edge_index = torch.tensor(np.array([edge_idx_arr_0, edge_idx_arr_1]), dtype=torch.long)

        graph = torch_geometric.data.Data(x=text_data, edge_index=edge_index, edge_attr=edge_feature_arr)

        graph_netx = torch_geometric.utils.to_networkx(graph, to_undirected=True)

        #pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

        options = {
            "font_size": 16,
            "node_size": 2000,
            "node_color": "red",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 5,
        }
        options2 = {
            "font_size": 0,
            "node_size": 8,
            "node_color": "red",
            "edgecolors": "black",
            "linewidths": 0,
            "width": 1,
        }

        nx.draw_networkx(graph_netx, **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

        torch.save(graph, 'graph.pt')

    if pargs.mode == 'gnn':

        model_name = "RoBERTa"

        features = np.load("./catalogues/{}/embeddings/ResNet/visual_embeddings.npy".format(dataset))
        text_embeddings = np.load("./catalogues/{}/embeddings/{}/text_embeddings.npy".format(dataset, model_name))
        k = 6

        adj = np.zeros([text_embeddings.shape[0], text_embeddings.shape[0]], dtype='float32')

        for product_idx in range(text_embeddings.shape[0]):
            distances_by_axis = (text_embeddings[product_idx,] - text_embeddings) ** 2
            distance_to_product = np.sqrt(distances_by_axis.sum(axis=1))
            k_nearest_idxs = distance_to_product.argsort()[1:k+1]
            k_nearest_distances = distance_to_product[k_nearest_idxs,]

            for n in range(k):
                adj[product_idx, k_nearest_idxs[n]] = 1

        ssearch.features = features
        ssearch.enable_search = True
        original_features = np.copy(ssearch.features)
        adjust_query = False
        metric = 'l2'
        norm = 'None'
        data_path = "./catalogues/{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        real_df = None
        eval_path = "./catalogues/{}/test".format(dataset)
        eval_files = ["./catalogues/{}/test/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

        new_visual_embeddings = train_cora(features, adj, gnn, None, 200, 0.01)
        #new_visual_embeddings = np.loadtxt("./visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')

        ssearch.features = new_visual_embeddings
        if pargs.list is None:
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            
            for fquery in eval_files:
                #print(fquery)
                im_query = ssearch.read_image(fquery)
                idx, dist_sorted, q_fv, data_search = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_features, df=df) 
                #print("idx: ", idx)              
                r_filenames = ssearch.get_filenames(idx)
                
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                ap, ap_tree, ap_sub = avg_precision(base_category, products)
                #print("{}: {}".format(fquery, ap))  
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                ap_arr_sub.append(ap_sub)

                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm)
                # output_name = os.path.join(pargs.odir, output_name)
                output_name = os.path.join("./catalogues/{}/results".format(dataset), output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))

            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP(GC): {}, mAP(CT): {}".format(mAP, mAP_tree))


    if pargs.mode == 'search' :
        ssearch.load_features()        
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)         
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)# 
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))             
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=99)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                
                # mAP
                #base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                #ap = avg_precision(base_category, products)
                #print("Average precision: {}".format(ap))  
                #####
                
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')
    if pargs.mode == 'search-umap':
        ssearch.load_features() 
        if pargs.umap:
            print("Training UMAP")
            n_neighbors = 15
            min_dist = 0.1
            n_components = 32
            n_epochs = 500
            vectors = np.asarray(ssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            reducer = umap.UMAP(n_components=n_components, n_epochs=n_epochs, random_state=42) #n_neighbors=15, min_dist=0.1
            reducer.fit(vectors)     
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)# 
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)  
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result_umap.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))             
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=99, reducer=reducer)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap = avg_precision(base_category, products)
                print(ap)  
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result_umap.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')

    

    if pargs.mode == 'eval-umap-clip' :

        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        
        #model_name = "clip-fn"
        #checkpoint_path = ".\CLIP_models/{}/model1.pt".format(dataset)
        #clipssearch = CLIPSSearch(pargs.config, pargs.name, checkpoint_path=checkpoint_path)
        
        data_path = "./catalogues/{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if "Pepeganga" in dataset:
            which_realq = "RealQ_2"
            real_df = pd.read_excel(".\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
        else:
            which_realq = "RealQ_3"
            real_df = pd.read_excel(".\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3

        
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.cosine_sim()
        clipssearch.load_features()

        pepeganga_rq_parameters = {"k_7_a_03_mean_4_dim_29_seed": (7, 0.3, None, False, 'mean', 4, 29),
                                    "k_7_a_03_mean_4_dim_27_seed": (7, 0.3, None, False, 'mean', 4, 27),
                                    "k_7_a_03_mean_4_dim_5_seed": (7, 0.3, None, False, 'mean', 4, 5),
                                    "k_7_a_03_mean_4_dim_18_seed": (7, 0.3, None, False, 'mean', 4, 18),
                                    "k_7_a_03_mean_4_dim_22_seed": (7, 0.3, None, False, 'mean', 4, 22),
                                    "k_7_a_03_mean_8_dim_29_seed": (7, 0.3, None, False, 'mean', 8, 29),
                                    "k_7_a_03_mean_8_dim_27_seed": (7, 0.3, None, False, 'mean', 8, 27),
                                    "k_7_a_03_mean_8_dim_5_seed": (7, 0.3, None, False, 'mean', 8, 5),
                                    "k_7_a_03_mean_8_dim_18_seed": (7, 0.3, None, False, 'mean', 8, 18),
                                    "k_7_a_03_mean_8_dim_22_seed": (7, 0.3, None, False, 'mean', 8, 22),
                                    "k_7_a_03_mean_16_dim_29_seed": (7, 0.3, None, False, 'mean', 16, 29),
                                    "k_7_a_03_mean_16_dim_27_seed": (7, 0.3, None, False, 'mean', 16, 27),
                                    "k_7_a_03_mean_16_dim_5_seed": (7, 0.3, None, False, 'mean', 16, 5),
                                    "k_7_a_03_mean_16_dim_18_seed": (7, 0.3, None, False, 'mean', 16, 18),
                                    "k_7_a_03_mean_16_dim_22_seed": (7, 0.3, None, False, 'mean', 16, 22),
                                    "k_7_a_03_mean_32_dim_29_seed": (7, 0.3, None, False, 'mean', 32, 29),
                                    "k_7_a_03_mean_32_dim_27_seed": (7, 0.3, None, False, 'mean', 32, 27),
                                    "k_7_a_03_mean_32_dim_5_seed": (7, 0.3, None, False, 'mean', 32, 5),
                                    "k_7_a_03_mean_32_dim_18_seed": (7, 0.3, None, False, 'mean', 32, 18),
                                    "k_7_a_03_mean_32_dim_22_seed": (7, 0.3, None, False, 'mean', 32, 22),
                                    "k_7_a_03_mean_64_dim_29_seed": (7, 0.3, None, False, 'mean', 64, 29),
                                    "k_7_a_03_mean_64_dim_27_seed": (7, 0.3, None, False, 'mean', 64, 27),
                                    "k_7_a_03_mean_64_dim_5_seed": (7, 0.3, None, False, 'mean', 64, 5),
                                    "k_7_a_03_mean_64_dim_18_seed": (7, 0.3, None, False, 'mean', 64, 18),
                                    "k_7_a_03_mean_64_dim_22_seed": (7, 0.3, None, False, 'mean', 64, 22),}

        homy_rq_parameters = {"k_3_t_0.5_softmax_4_dim_29_seed": (3, None, 0.5, True, 'softmax', 4, 29),
                            "k_3_t_0.5_softmax_4_dim_27_seed": (3, None, 0.5, True, 'softmax', 4, 27),
                            "k_3_t_0.5_softmax_4_dim_5_seed": (3, None, 0.5, True, 'softmax', 4, 5),
                            "k_3_t_0.5_softmax_4_dim_18_seed": (3, None, 0.5, True, 'softmax', 4, 18),
                            "k_3_t_0.5_softmax_4_dim_22_seed": (3, None, 0.5, True, 'softmax', 4, 22),
                            "k_3_t_0.5_softmax_8_dim_29_seed": (3, None, 0.5, True, 'softmax', 8, 29),
                            "k_3_t_0.5_softmax_8_dim_27_seed": (3, None, 0.5, True, 'softmax', 8, 27),
                            "k_3_t_0.5_softmax_8_dim_5_seed": (3, None, 0.5, True, 'softmax', 8, 5),
                            "k_3_t_0.5_softmax_8_dim_18_seed": (3, None, 0.5, True, 'softmax', 8, 18),
                            "k_3_t_0.5_softmax_8_dim_22_seed": (3, None, 0.5, True, 'softmax', 8, 22),
                            "k_3_t_0.5_softmax_16_dim_29_seed": (3, None, 0.5, True, 'softmax', 16, 29),
                            "k_3_t_0.5_softmax_16_dim_27_seed": (3, None, 0.5, True, 'softmax', 16, 27),
                            "k_3_t_0.5_softmax_16_dim_5_seed": (3, None, 0.5, True, 'softmax', 16, 5),
                            "k_3_t_0.5_softmax_16_dim_18_seed": (3, None, 0.5, True, 'softmax', 16, 18),
                            "k_3_t_0.5_softmax_16_dim_22_seed": (3, None, 0.5, True, 'softmax', 16, 22),
                            "k_3_t_0.5_softmax_32_dim_29_seed": (3, None, 0.5, True, 'softmax', 32, 29),
                            "k_3_t_0.5_softmax_32_dim_27_seed": (3, None, 0.5, True, 'softmax', 32, 27),
                            "k_3_t_0.5_softmax_32_dim_5_seed": (3, None, 0.5, True, 'softmax', 32, 5),
                            "k_3_t_0.5_softmax_32_dim_18_seed": (3, None, 0.5, True, 'softmax', 32, 18),
                            "k_3_t_0.5_softmax_32_dim_22_seed": (3, None, 0.5, True, 'softmax', 32, 22),
                            "k_3_t_0.5_softmax_64_dim_29_seed": (3, None, 0.5, True, 'softmax', 64, 29),
                            "k_3_t_0.5_softmax_64_dim_27_seed": (3, None, 0.5, True, 'softmax', 64, 27),
                            "k_3_t_0.5_softmax_64_dim_5_seed": (3, None, 0.5, True, 'softmax', 64, 5),
                            "k_3_t_0.5_softmax_64_dim_18_seed": (3, None, 0.5, True, 'softmax', 64, 18),
                            "k_3_t_0.5_softmax_64_dim_22_seed": (3, None, 0.5, True, 'softmax', 64, 22),}
        

        top=20
        adjust_query = True
        adjust_query_sim = False
        metric = 'l2'
        norm = 'None'
        
        original_features = np.copy(clipssearch.features)
        new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=3, a=None, T=0.5, use_query=True, method='softmax') ## HOMY
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=7, a=0.3, T=None, use_query=False, method='mean') ## PEPEGANGA
        #new_visual_embeddings = np.loadtxt("./visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
        clipssearch.features = new_visual_embeddings



        # USE OR NOT REAL QUERIES FOR EVALUATION
        eval_path = "./visual_attributes/{}/eval_images".format(which_realq)
        eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        eval_data = []
        for k, params in pepeganga_rq_parameters.items():
        
            ap_arr = []
            ap_arr_tree = []

            reducer = None
            print(f"Training UMAP: {params[5]}-dim, seed {params[6]}")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = params[5]
            vectors = np.asarray(clipssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            #42
            reducer = umap.UMAP(n_components=n_components, random_state=params[6]) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
            print("Done training UMAP")

            for fquery in eval_files:
                #print(fquery)
                im_query = clipssearch.read_image(fquery)

                #idx = ssearch.search(im_query, metric, top=20)
                idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, adjust_query_sim=adjust_query_sim, original_embeddings=original_features, df=data_df, reducer=reducer, text_model=textSearch)                
                r_filenames = clipssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                #r_filenames_top_20 = r_filenames[:21]  # Top 20
                base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                ap, ap_tree, _ = avg_precision(base_category, products, use_all_categories=True)
                #print("{}: {}".format(fquery, ap))  
                #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            eval_data.append([k, mAP, mAP_tree, params[5], params[6]])
            
        df = pd.DataFrame(eval_data, columns=['params', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'UMAP dim', 'seed'])
        df_path = "./{}/UMAP/gc_3_no_q.xlsx".format(dataset)
        #pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
        df.to_excel(df_path, index=False)



        #########################################

    if pargs.mode == 'eval-umap' :

        ##### Only when using text
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name="word2vec")
        #textSearch.set_gensim_model() # word2vec
        #textSearch.set_dataframes()
        #textSearch.cosine_sim()
        # Text Visual Parameters
        k = 5
        alpha = 0.6
        T = None
        use_query = False
        method = 'mean'
        #######################
        ssearch.load_features()

        # When using text
        #new_visual_embeddings = textSearch.adjust_visual_embeddings(ssearch.features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
        #ssearch.features = new_visual_embeddings

        eval_path = "./{}/eval_images".format(dataset)
        eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        if True:
            print("Training UMAP")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = 16
            n_epochs = 500
            vectors = np.asarray(ssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_epochs=n_epochs) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
        if pargs.list is None: #### IS NONE
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            i = 0
            for fquery in eval_files:
                i += 1
                print(fquery)
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, top=20, reducer=reducer)           
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                print("Average precision for {}: {}".format(fquery, ap)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                ap_arr_sub.append(ap_sub)
                #image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                #output_name = os.path.join('./results_umap', output_name)
                #io.imsave(output_name, image_r)
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                #if i == 10:
                #    break
            
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP global: {}".format(mAP))
            print("mAP tree: {}".format(mAP_tree))
            print("mAP sub: {}".format(mAP_sub))




    if pargs.mode == 'eval-all-umap' :

        ##### Only when using text
        #model_name="roberta"
        #textSearch = textsearch.TextSearch(model_name=model_name, filter_by_nwords=False, build=True, dataset=dataset)
        #textSearch.set_model()
        #textSearch.set_dataframes()
        #textSearch.cosine_sim()

        data_path = "./catalogues/{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        # Text Visual Parameters

        use_multiple_seeds = True

        this_parameters = {"base_64_dim_29": (None, None, None, None, 'base', 64, 29),
                            "base_64_dim_27": (None, None, None, None, 'base', 64, 27),
                            "base_64_dim_5": (None, None, None, None, 'base', 64, 5),
                            "base_64_dim_18": (None, None, None, None, 'base', 64, 18),
                            "base_64_dim_22": (None, None, None, None, 'base', 64, 22),
                            "base_32_dim_29": (None, None, None, None, 'base', 32, 29),
                            "base_32_dim_27": (None, None, None, None, 'base', 32, 27),
                            "base_32_dim_5": (None, None, None, None, 'base', 32, 5),
                            "base_32_dim_18": (None, None, None, None, 'base', 32, 18),
                            "base_32_dim_22": (None, None, None, None, 'base', 32, 22),
                        "base_16_dim_29": (None, None, None, None, 'base', 16, 29),
                        "base_dim_27": (None, None, None, None, 'base', 16, 27),
                        "base_16_dim_5": (None, None, None, None, 'base', 16, 5),
                        "base_16_dim_18": (None, None, None, None, 'base', 16, 18),
                        "base_16_dim_22": (None, None, None, None, 'base', 16, 22),
                        "base_8_dim_29": (None, None, None, None, 'base', 8, 29),
                        "base_8_dim_27": (None, None, None, None, 'base', 8, 27),
                        "base_8_dim_5": (None, None, None, None, 'base', 8, 5),
                        "base_8_dim_18": (None, None, None, None, 'base', 8, 18),
                        "base_8_dim_22": (None, None, None, None, 'base', 8, 22),
                        "base_4_dim_29": (None, None, None, None, 'base', 4, 29),
                        "base_4_dim_27": (None, None, None, None, 'base', 4, 27),
                        "base_4_dim_5": (None, None, None, None, 'base', 4, 5),
                        "base_4_dim_18": (None, None, None, None, 'base', 4, 18),
                        "base_4_dim_22": (None, None, None, None, 'base', 4, 22),
                        }

        #######################
        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        clipssearch.load_features()

        top=20
        adjust_query = False
        adjust_query_sim = False
        metric = 'l2'
        norm = 'None'

        adjusted_embeddings = False
        original_features = np.copy(clipssearch.features)
        
        new_visual_embeddings = None
        #self, embeddings, filenames, k=3, method='mean', a=0.9, T=1.0, use_query=True
        # mean: k, a, T, use_query=False, method='mean'
        eval_data = []
        # mean: k, a, T, use_query=False, method='mean'
        params_set = this_parameters

        which_realq = "RealQ_3"

        if "Pepeganga" in dataset:
            real_df = pd.read_excel(".\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
        else:
            real_df = pd.read_excel(".\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3

        for k, params in params_set.items():
            ap_arr = []
            ap_arr_tree = []

            reducer = None
            print(f"Training UMAP: {params[5]}-dim, seed {params[6]}")
            #n_neighbors = 15
            #min_dist = 0.1
            n_components = params[5]
            vectors = np.asarray(clipssearch.features)
            #print("----------------LARGO DE FEATURES: ", len(vectors))
            #42
            reducer = umap.UMAP(n_components=n_components, random_state=params[6]) #n_neighbors=n_neighbors, min_dist=min_dist
            reducer.fit(vectors)
            print("Done training UMAP")

            # USE OR NOT REAL QUERIES FOR EVALUATION
            eval_path = "./{}/eval_images".format(which_realq)
            eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
            for fquery in eval_files:
                #print(fquery)
                im_query = clipssearch.read_image(fquery)
                idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=20, reducer=reducer, original_embeddings=original_features)           
                r_filenames = clipssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                ap, ap_tree, _ = avg_precision(base_category, products, use_all_categories=True)
                #print("Average precision for {}: {}".format(fquery, ap)) 
                ap_arr.append(ap)
                ap_arr_tree.append(ap_tree)
                #image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                #output_name = os.path.join('./results_umap', output_name)
                #io.imsave(output_name, image_r)
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
            
            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            #mAP_sub = statistics.mean(ap_arr_sub)
            eval_data.append([k, n_components, mAP, mAP_tree])
            print("mAP global: {}".format(mAP))
            print("mAP tree: {}".format(mAP_tree))
            #print("mAP sub: {}".format(mAP_sub))
        df = pd.DataFrame(eval_data, columns=['params', 'UMAP dim', 'mAP (GlobalCategory)', 'mAP (CategoryTree)'])
        df_path = "./{}/UMAP/baseline/gc_3_no_q.xlsx".format(dataset)
        df.to_excel(df_path, index=False)
        
    if pargs.mode == 'eval-text':
        eval_data = []
        #all_components = [512, 256, 128, 64, 32, 16, 8]
        #seeds = [29, 27, 5, 18, 22]
        all_components = [8]
        seeds = [1]
        #all_components = [16, 8]
        for components in all_components:
            for seed in seeds:
                textSearch = textsearch.TextSearch(model_name="clip-base", build=False, dataset=dataset, filter_by_nwords=False)
                #textSearch.set_dataframes()
                textSearch.set_model() # word2vec
                m_ap, m_ap_tree, m_ap_sub = textSearch.eval_model(k=20, use_umap=False, n_components=components, seed=seed)
                eval_data.append([seed, components, m_ap, m_ap_tree, m_ap_sub])
                print("-----------")
                print("mAP (GC): {}".format(m_ap))
                print("-----------")
                print("mAP (CT): {}".format(m_ap_tree))
                print("-----------")
                print("mAP (SC): {}".format(m_ap_sub))
        df = pd.DataFrame(eval_data, columns=['seed', 'UMAP dim', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'mAP (SubCategory)'])
        #df.to_excel("./visual_text_embeddings_results/{}/results_text_umap_500.xlsx".format(dataset))

    if pargs.mode == 'search-text-visual':
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset)
        textSearch.set_gensim_model() # word2vec
        textSearch.set_dataframes()
        ssearch.load_features()

        # PARAMS
        k = 7
        alpha = 0.3
        #k = 3
        #alpha = 0.5
        T = None
        use_query = False
        #method = "mean"
        method = "base"
        filename = "base"
        #filename = "k_5_a_02"

        print("Going to adjust visual embeddings")
        if method != "base":
            new_visual_embeddings = textSearch.adjust_visual_embeddings(ssearch.features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
            #new_visual_embeddings = np.loadtxt("./visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
            ssearch.features = new_visual_embeddings
        else:
            print("Searching only with visual embeddings.")

        if pargs.list is None:
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                #idx = ssearch.search(im_query, metric, top=99)
                norm = "None"
                metric='l2'
                idx = ssearch.search(im_query, metric=metric, top=99, norm=norm)

                r_filenames = ssearch.get_filenames(idx)            
                
                r_filenames.insert(0, fquery)

                # Evaluation
                #base_category, products = get_product_and_category(r_filenames, dataset=dataset)
                #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                #print("AP (GC): {}\nAP (CT): {}\nAP (SC): {}".format(ap, ap_tree, ap_sub))  
                #############

                image_r= ssearch.draw_result(r_filenames)
                #output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(filename, metric, norm)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
                fquery = input('Query:')

        #ap_arr, m_ap, res = textSearch.search_product('Peluche Asia 44 cm')
        #print("Average Precision array:")
        #pprint(ap_arr)
        #print("-----------")
        #print("mAP: {}".format(m_ap))
        #print("-----------")
    
    if pargs.mode == 'eval-all-text-visual':
        model_name = 'word2vec' ######################
        textSearch = textsearch.TextSearch(model_name=model_name, filter_by_nwords=False, build=False, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.cosine_sim()
        ssearch.load_features()
        original_features = np.copy(ssearch.features)
        adjust_query = False
        metric = 'l2'
        norm = 'None'
        data_path = "./catalogues/{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")
        if use_real_queries:
            if "Pepeganga" in dataset:
                which_realq = "RealQ_2"
                real_df = pd.read_excel(".\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                which_realq = "RealQ_3"
                real_df = pd.read_excel(".\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
            eval_path = "./{}/eval_images".format(which_realq)
            eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        else:
            real_df = None
            eval_path = "./catalogues/{}/test".format(dataset)
            eval_files = ["./catalogues/{}/test/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        eval_data = []
        params_set = parameters
        for k, params in params_set.items():
            if k != "Base":
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, ssearch.filenames, k=params[0], a=params[1], T=params[2], use_query=params[3], method=params[4])
                #new_visual_embeddings = np.loadtxt("./visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                ssearch.features = new_visual_embeddings
            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []
                
                for fquery in eval_files:
                    #print(fquery)
                    im_query = ssearch.read_image(fquery)
                    idx, dist_sorted, q_fv, data_ssearch = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_features, df=df) 
                    #print("idx: ", idx)              
                    r_filenames = ssearch.get_filenames(idx)
                    
                    r_filenames.insert(0, fquery)
                    base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                    # ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    ap, ap_tree, ap_sub = avg_precision(base_category, products)
                    #print("{}: {}".format(fquery, ap))  
                    ap_arr.append(ap)
                    ap_arr_tree.append(ap_tree)
                    ap_arr_sub.append(ap_sub)

                mAP = statistics.mean(ap_arr)
                mAP_tree = statistics.mean(ap_arr_tree)
                mAP_sub = statistics.mean(ap_arr_sub)
                eval_data.append([k, mAP, mAP_tree, mAP_sub])
                print("params: {}, mAP(GC): {}, mAP(CT): {}".format(k, mAP, mAP_tree))

        df = pd.DataFrame(eval_data, columns=['params', 'mAP (GlobalCategory)', 'mAP (CategoryTree)', 'mAP (SubCategory)'])
        df.to_excel("./visual_text_embeddings_results/{}/{}/all_metrics_catalog_{}_{}_queryadj_{}.xlsx".format(dataset, textSearch.get_model_name(), metric, norm, adjust_query))

    if pargs.mode == 'eval-text-visual':
        top=99
        model_name = "roberta"
        data_path = "./catalogues/{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if use_real_queries:
            if "Pepeganga" in dataset:
                real_df = pd.read_excel(".\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
                real_queries_path = "real_queries_2"
                which_realq = "RealQ_2"
            else:
                real_df = pd.read_excel(".\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
                real_queries_path = "real_queries"
                which_realq = "RealQ_3"   
            base_path = "./{}/results_visual_text/{}".format(dataset, real_queries_path)
            eval_path = "./{}/eval_images".format(which_realq)
            eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        else:
            real_df = None
            base_path = "./{}/results_visual_text".format(dataset)
            eval_path = "./{}/eval_images".format(dataset)
            eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.cosine_sim()
        ssearch.load_features()
        original_features = np.copy(ssearch.features)
        
        params = Pepeganga_RQ_params
        
        for category_metric, best_params in params.items():
            eval_data = []
            if 'adj' in category_metric:
                adjust_query = True
            else:
                adjust_query = False

            k=best_params[0]
            alpha=best_params[1]
            T=best_params[2]
            use_query=best_params[3]
            method=best_params[4]
            metric = 'l2'
            norm = 'None'
            
            #######################

            if method == 'mean':
                if use_query:    
                    #generated_file = "k_{}_a_{}_{}.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = 'k_{}_a_{}_{}'.format(str(k), str(alpha).replace(".", ""), method)
                else:
                    #generated_file = "k_{}_a_{}_{}_noquery.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = "k_{}_a_{}_{}_noquery".format(str(k), str(alpha).replace(".", ""), method)
            elif method == 'sim':
                #use_query = True
                #generated_file = "k_{}_{}.txt".format(str(k), method)
                results_dir = "k_{}_{}".format(str(k), method)
            elif method == 'softmax':
                #use_query = True
                #generated_file = "k_{}_t_{}_{}.txt".format(str(k), T, method)
                results_dir = "k_{}_t_{}_{}".format(str(k), T, method)
            elif method == 'base':
                results_dir = "base"
            results_path = "{}/{}_{}_{}_queryadj_{}".format(base_path, results_dir, metric, norm, adjust_query)
            pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
            ########################################

            original_embeddings = ssearch.features
            if method != 'base':
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, ssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
                ssearch.features = new_visual_embeddings

            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []

            for i, fquery in enumerate(eval_files):
                #print(fquery)
                im_query = ssearch.read_image(fquery)

                #idx = ssearch.search(im_query, metric, top=20)
                # idx_sorted, d_sorted, q_fv, data
                idx, dist_array, residualvar1, residualvar2 = ssearch.search(im_query=im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_features, df=data_df)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                #r_filenames_top_20 = r_filenames[:21]  # Top 20
                #base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                #print("{}: {}".format(fquery, ap))  
                #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                #ap_arr.append(ap)
                #ap_arr_tree.append(ap_tree)
                #ap_arr_sub.append(ap_sub)
                
                image_r= ssearch.draw_result(r_filenames) # Not writing distance
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)

                if use_real_queries:
                    output_name = os.path.join('./{}/results_visual_text/{}/{}_{}_{}_queryadj_{}'.format(dataset, real_queries_path, results_dir, metric, norm, adjust_query), output_name)
                else:
                    output_name = os.path.join('./{}/results_visual_text/{}_{}_{}_queryadj_{}'.format(dataset, results_dir, metric, norm, adjust_query), output_name)
                    
                io.imsave(output_name, image_r)

                #eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])

                #if i == 2:
                #    break
                
                #print('result saved at {}'.format(output_name))
                #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                #print("r_filenames: {}\n".format(r_filenames))  
            
            #df = pd.DataFrame(eval_data, columns=['fquery',  'AP (GC)', 'AP (CT)', 'AP (SC)'])
            #if use_real_queries:
            #    df_path = "./visual_text_embeddings_results/{}/{}/{}".format(dataset, textSearch.get_model_name(), real_queries_path)
            #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
            #else:
            #    df_path = "./visual_text_embeddings_results/{}/{}".format(dataset, textSearch.get_model_name())
            #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
            #df.to_excel("{}/{}_ap_20_{}_{}_queryadj_{}.xlsx".format(df_path, results_dir, metric, norm, adjust_query), index=False)


            #mAP = statistics.mean(ap_arr)
            #mAP_tree = statistics.mean(ap_arr_tree)
            #mAP_sub = statistics.mean(ap_arr_sub)
            #print("mAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(mAP, mAP_tree, mAP_sub))


    if pargs.mode == "eval-text-visual-clip":
        
        top=99

        model_name = "clip-base"
        clipssearch = CLIPSSearch(pargs.config, pargs.name) #BASE
        
        data_path = "./catalogues/{}/data/".format(dataset)
        data_df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        if use_real_queries:
            if "Pepeganga" in dataset:
                real_queries_path = "real_queries_2"
                which_realq = "RealQ_2"
                real_df = pd.read_excel(".\Datasets\Pepeganga_GT/realqueries_2_info.xlsx") # REALQ2
            else:
                real_queries_path = "real_queries"
                which_realq = "RealQ_3"
                real_df = pd.read_excel(".\Datasets\Catalogo_Homyold/real_queries/queries.xlsx") # REALQ3
            
            base_path = "./{}/results_visual_text/{}".format(dataset, real_queries_path)
            eval_path = "./{}/eval_images".format(which_realq)
            eval_files = ["{}/eval_images/".format(which_realq) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        else:
            real_df = None
            base_path = "./{}/results_visual_text".format(dataset)
            eval_path = "./{}/eval_images".format(dataset)
            eval_files = ["{}/eval_images/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]
        
        textSearch = textsearch.TextSearch(filter_by_nwords=False, build=False, dataset=dataset, model_name=model_name)
        #textSearch = textsearch.TextSearch(filter_by_nwords=False, build=True, dataset=dataset)
        textSearch.set_model()
        textSearch.set_dataframes()
        textSearch.cosine_sim()
        clipssearch.load_features()
        original_features = np.copy(clipssearch.features)
        
        params = HomyCLIPTrained_params
        
        for category_metric, best_params in params.items():
            eval_data = []
            if 'adj' in category_metric:
                #print("adj in category metric")
                adjust_query = True
            else:
                #print("adj NOT in category metric")
                adjust_query = False

            k=best_params[0]
            alpha=best_params[1]
            T=best_params[2]
            use_query=best_params[3]
            method=best_params[4]
            metric = 'l2'
            norm = 'None'
            
            #######################

            if method == 'mean':
                if use_query:    
                    #generated_file = "k_{}_a_{}_{}.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = 'k_{}_a_{}_{}'.format(str(k), str(alpha).replace(".", ""), method)
                else:
                    #generated_file = "k_{}_a_{}_{}_noquery.txt".format(str(k), str(a).replace(".", ""), method)
                    results_dir = "k_{}_a_{}_{}_noquery".format(str(k), str(alpha).replace(".", ""), method)
            elif method == 'sim':
                #use_query = True
                #generated_file = "k_{}_{}.txt".format(str(k), method)
                results_dir = "k_{}_{}".format(str(k), method)
            elif method == 'softmax':
                #use_query = True
                #generated_file = "k_{}_t_{}_{}.txt".format(str(k), T, method)
                results_dir = "k_{}_t_{}_{}".format(str(k), T, method)
            elif method == 'base':
                results_dir = "base"
            results_path = "{}/{}_{}_{}_queryadj_{}".format(base_path, results_dir, metric, norm, adjust_query)
            pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
            ########################################

            original_embeddings = clipssearch.features
            if method != 'base':
                new_visual_embeddings = textSearch.adjust_visual_embeddings(original_features, clipssearch.filenames, k=k, a=alpha, T=T, use_query=use_query, method=method)
                #new_visual_embeddings = np.loadtxt("./visual_text_embeddings/{}/test.txt".format(dataset), delimiter='\t')
                clipssearch.features = new_visual_embeddings


            if pargs.list is None:
                ap_arr = []
                ap_arr_tree = []
                ap_arr_sub = []

                for i, fquery in enumerate(eval_files):
                    #print(fquery)
                    im_query = clipssearch.read_image(fquery)

                    #idx = ssearch.search(im_query, metric, top=20)
                    idx, dist_array = clipssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_features, df=data_df)                
                    r_filenames = clipssearch.get_filenames(idx)
                    r_filenames.insert(0, fquery)
                    #r_filenames_top_20 = r_filenames[:21]  # Top 20
                    #base_category, products = get_product_and_category(r_filenames, dataframe=data_df, real_df=real_df)
                    #ap, ap_tree, ap_sub = avg_precision(base_category, products, use_all_categories=True)
                    #print("{}: {}".format(fquery, ap))  
                    #print("Average precision for {}: {} (GC), {} (CT)".format(fquery, ap, ap_tree)) 
                    #ap_arr.append(ap)
                    #ap_arr_tree.append(ap_tree)
                    #ap_arr_sub.append(ap_sub)
                    
                    image_r= ssearch.draw_result(r_filenames) # Not writing distance
                    output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)

                    if use_real_queries:
                        output_name = os.path.join('./{}/results_visual_text/{}/{}_{}_{}_queryadj_{}'.format(dataset, real_queries_path, results_dir, metric, norm, adjust_query), output_name)
                    else:
                        output_name = os.path.join('./{}/results_visual_text/{}_{}_{}_queryadj_{}'.format(dataset, results_dir, metric, norm, adjust_query), output_name)
                        
                    io.imsave(output_name, image_r)

                    #eval_data.append([os.path.splitext(os.path.basename(fquery))[0], ap, ap_tree, ap_sub])

                    #if i == 2:
                    #    break
                    
                    #print('result saved at {}'.format(output_name))
                    #print("Largo de r_filenames: {}\n".format(len(r_filenames)))     
                    #print("r_filenames: {}\n".format(r_filenames))  
                
                #df = pd.DataFrame(eval_data, columns=['fquery',  'AP (GC)', 'AP (CT)', 'AP (SC)'])
                #if use_real_queries:
                #    df_path = "./visual_text_embeddings_results/{}/{}/{}".format(dataset, textSearch.get_model_name(), real_queries_path)
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #else:
                #    df_path = "./visual_text_embeddings_results/{}/{}".format(dataset, textSearch.get_model_name())
                #    pathlib.Path(df_path).mkdir(parents=False, exist_ok=True)
                #df.to_excel("{}/{}_ap_20_{}_{}_queryadj_{}.xlsx".format(df_path, results_dir, metric, norm, adjust_query), index=False)


                #mAP = statistics.mean(ap_arr)
                #mAP_tree = statistics.mean(ap_arr_tree)
                #mAP_sub = statistics.mean(ap_arr_sub)
                #print("mAP (GC): {}\nmAP (CT): {}\nmAP (SC): {}".format(mAP, mAP_tree, mAP_sub))

    if pargs.mode == 'utils':
        ssearch.load_features()
        prepare_dataset(dataset, ssearch.features, ssearch.filenames, use_umap=True, n_components=128)     
