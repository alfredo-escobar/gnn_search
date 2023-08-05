from ntpath import join
import sys
from scipy import spatial
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
import umap.plot
import torch
from visual_text_parameters import parameters
from data_utils import prepare_dataset
from bpm_parameters import *
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from clip_ssearch import CLIPSSearch
from pprint import pprint
from scipy.spatial.distance import cdist

from save_sims import most_similar_pairs


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

 
    def search(self, im_query, metric = 'l2', norm = 'None', top=90, reducer=None, vtnn=None, adjust_query=False, adjust_query_sim=False, original_embeddings=None, df=None, text_model=None, query_is_feature=False):
        assert self.enable_search, 'search is not allowed'

        if query_is_feature:
            q_fv = im_query
        else:
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
            data = None
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
    
    def set_features(self, features):
        self.features = features
                    

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


def plot_loss_and_mAP(historical_loss, historical_mAP):
    x_loss = range(len(historical_loss))
    
    # step_mAP = int(len(historical_loss) / len(historical_mAP))
    # x_mAP  = [-1, *range(0, len(historical_mAP)*2, step_mAP)]
    # x_mAP = x_mAP[0:len(historical_mAP)]

    x_mAP = range(len(historical_mAP))
    fig = plt.figure()

    # Top subplot (2 rows, 1 col, subplot 1)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_loss, historical_loss)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss in training')

    # Bottom subplot (2 rows, 1 col, subplot 2)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_mAP, historical_mAP)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('mAP')
    ax2.set_title('mAP in training')

    fig.subplots_adjust(hspace=0.4)
    plt.show()


def plot_histogram(similarity_top, random_indexes_similar, random_indexes_dissimilar):
    num_bins = 50

    fig, ax = plt.subplots()

    x_multi = [similarity_top,
               similarity_top[random_indexes_similar],
               similarity_top[random_indexes_dissimilar]]
    
    colors = ['blue', 'green', 'red']

    ax.hist(x_multi, num_bins, histtype='bar', color=colors, label=colors)

    ax.set_xlabel('Similarity score')
    ax.set_ylabel('Frequency')
    plt.legend(loc="upper right")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def get_current_mAP(current_embeddings,
                    ssearch,
                    eval_files,
                    metric,
                    norm,
                    adjust_query,
                    original_embeddings,
                    df,
                    real_df):
    
    ssearch.features = current_embeddings
    
    ap_arr = []
    ap_arr_tree = []
    ap_arr_sub = []

    print("Computing mAP")

    for fquery in eval_files:
        im_query = ssearch.read_image(fquery)
        idx, dist_sorted, q_fv, data_search = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_embeddings, df=df)         
        r_filenames = ssearch.get_filenames(idx)
        
        r_filenames.insert(0, fquery)
        base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
        ap, ap_tree, ap_sub = avg_precision(base_category, products)
        ap_arr.append(ap)
        ap_arr_tree.append(ap_tree)
        ap_arr_sub.append(ap_sub)

    mAP = statistics.mean(ap_arr)
    mAP_tree = statistics.mean(ap_arr_tree)
    mAP_sub = statistics.mean(ap_arr_sub)
    print("mAP(GC): {}, mAP(CT): {}".format(mAP, mAP_tree))

    return mAP


def reorder_text_embeddings_old(text_embeddings, catalog):
    # Necessary for both the visual and text embeddings
    # to be in the same order
    with open(catalog) as f_in:
        catalog_list = [item.lower() for item in f_in]
    indices = np.argsort(catalog_list)
    reordered_text_embeddings = text_embeddings[indices]
    return reordered_text_embeddings


def reorder_text_embeddings(text_embeddings, ve_catalog, te_catalog):

    with open(ve_catalog, 'r') as file_ve, open(te_catalog, 'r') as file_te:
        lines_ve = file_ve.read().splitlines()
        lines_te = file_te.read().splitlines()

    indices = np.zeros(len(lines_ve), dtype=np.int32)

    for i, element in enumerate(lines_ve):
        indices[i] = lines_te.index(element[0:-4])

    reordered_text_embeddings = text_embeddings[indices]
    return reordered_text_embeddings


def get_k_random_pairs(similarity, k = 100):

    # 2D indexes for the similarity matrix:
    similarity_idx = np.triu_indices(similarity.shape[0])

    # 1D array with similarity scores of upper half of matrix
    similarity_top = similarity[similarity_idx]

    # 1D indexes for the 2D indexes for the similarity matrix:
    indexes = np.arange(similarity_idx[0].shape[0])
    size = int(k/2)

    probs_sim    = similarity_top / np.sum(similarity_top)
    probs_dissim = (1.0 - similarity_top) / np.sum(1.0 - similarity_top)

    # k random 1D indexes for the 2D indexes for the similarity matrix:
    random_indexes_similar    = np.random.choice(indexes, size, replace=False, p=probs_sim)
    random_indexes_dissimilar = np.random.choice(indexes, size, replace=False, p=probs_dissim)

    # plot_histogram(similarity_top, random_indexes_similar, random_indexes_dissimilar)
    
    batch_mask = np.zeros((similarity.shape[0], similarity.shape[0]), dtype='float32')
    
    batch_mask[similarity_idx[0][random_indexes_similar],    similarity_idx[1][random_indexes_similar]] = 1.0
    batch_mask[similarity_idx[0][random_indexes_dissimilar], similarity_idx[1][random_indexes_dissimilar]] = 1.0

    return tf.convert_to_tensor(batch_mask)


def reduce_range_to_0_to_1(sim_text, sim_visual, margin=0.1):

    min_sim_text   = tf.math.reduce_min(sim_text)
    min_sim_visual = tf.math.reduce_min(sim_visual)
    min_overall = tf.math.minimum(min_sim_text, min_sim_visual)

    new_sim_text = sim_text - min_overall
    new_sim_visual = sim_visual - min_overall

    max_sim_text   = tf.math.reduce_max(new_sim_text)
    max_sim_visual = tf.math.reduce_max(new_sim_visual)
    max_overall = tf.math.maximum(max_sim_text, max_sim_visual)

    
    new_sim_text = new_sim_text / max_overall
    new_sim_text *= (1 - 2*margin)
    new_sim_text += margin

    
    new_sim_visual = new_sim_visual / max_overall
    new_sim_visual *= (1 - 2*margin)
    new_sim_visual += margin

    return new_sim_text, new_sim_visual


# def get_euclidean_distances(embeddings):

#     print("---Computing euclidean distances...---")

#     size = embeddings.shape[0]

#     distances_list = []

#     for idx in range(size):
#         row = tf.expand_dims(embeddings[idx], axis=0)
#         tiled_row = tf.tile(row, multiples=(size, 1)) # same shape as "embeddings", but
#                                                       # all of the rows are embeddings[idx]
#         diff_per_dim = tf.subtract(tiled_row, embeddings)
#         diff_per_dim = tf.square(diff_per_dim)

#         distances_to_current_row = tf.reduce_sum(diff_per_dim, axis=1)
#         distances_to_current_row = tf.sqrt(distances_to_current_row)

#         distances_list.append(distances_to_current_row)

#     distances = tf.stack(distances_list)

#     return distances


def get_euclidean_distances(embeddings):

    print("---Computing euclidean distances...---")

    size = embeddings.shape[0]

    distances_list = []

    for idx in range(size):

        diff_per_dim = tf.subtract(embeddings[idx], embeddings)

        dists_from_this = tf.norm(diff_per_dim, ord='euclidean', axis=1)

        distances_list.append(dists_from_this)

    distances = tf.stack(distances_list)

    return distances


def make_symmetrical(similarity):
    similarity_transposed = tf.transpose(similarity)
    multiplied_sim_simt = tf.multiply(similarity, similarity_transposed)

    similarity = tf.add(similarity, similarity_transposed)
    similarity = tf.subtract(similarity, multiplied_sim_simt)

    return similarity


def get_cos_similarity_tensor(embeddings):
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    return similarity


def get_cos_softmax_similarity_tensor(embeddings):
    # We get simiarity by cosine distance
    similarity = get_cos_similarity_tensor(embeddings)

    similarity = tf.nn.softmax(similarity)
    # Up until this point, all values in each row add up to 1
    # The similarity tensor became non symmetrical

    similarity = make_symmetrical(similarity)
    # Now, the similarity tensor is symmetrical again        

    return similarity


def get_sqrt_similarity_tensor(embeddings):
    distances = get_euclidean_distances(embeddings)

    similarity_denom = tf.exp(-distances)
    similarity = tf.divide(1.0, similarity_denom)

    similarity = tf.nn.softmax(similarity)
    similarity = make_symmetrical(similarity)

    return similarity


def get_sqrt_normmin_similarity_tensor(embeddings):
    distances = get_euclidean_distances(embeddings)
    min_distances = tf.expand_dims(tf.reduce_min(distances, axis=1), axis=1)

    similarity_denom = tf.exp(-(distances-min_distances))
    similarity = tf.divide(1.0, similarity_denom)

    similarity = tf.nn.softmax(similarity)
    similarity = make_symmetrical(similarity)

    return similarity


def gnn(fts, adj, transform, activation):
    seq_fts = transform(fts)
    ret_fts = tf.matmul(adj, seq_fts)
    return activation(ret_fts)


def loss_by_visual_text_contrast(similarity_visual, similarity_text, batch_mask, tau=1):

    dissimilarity_text = tf.subtract(1.0, similarity_text)
    dissimilarity_visual = tf.subtract(1.0, similarity_visual)

    similarity_weighted = tf.divide(similarity_text, similarity_visual)
    dissimilarity_weighted = tf.divide(dissimilarity_text, dissimilarity_visual)

    similars_addend = tf.multiply(similarity_text, tf.math.log(similarity_weighted))
    dissimilars_addend = tf.multiply(dissimilarity_text, tf.math.log(dissimilarity_weighted))

    loss_for_each_pair = tf.add(similars_addend, dissimilars_addend)
    loss_for_each_pair = tf.multiply(loss_for_each_pair, batch_mask)

    loss_aggregate = tf.divide(tf.reduce_sum(loss_for_each_pair), tf.reduce_sum(batch_mask))

    return loss_aggregate


def train_visual(visual_embeddings, text_embeddings, iterations, lr, mAP_dictionary = None):
    #mAP_dictionary = None

    adjust_range = True

    similarity_func = get_cos_similarity_tensor
    #similarity_func = get_cos_softmax_similarity_tensor
    #similarity_func = get_sqrt_similarity_tensor
    #similarity_func = get_sqrt_normmin_similarity_tensor

    similarity_text = similarity_func(text_embeddings)

    # Fines de testeo:
    #save_most_similar_pairs(similarity_text.numpy())

    units = visual_embeddings.shape[1]
    lyr = tf.keras.layers.Dense(units)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    historical_loss = []
    historical_mAP = []
    historical_mAP_maxsize = 5

    if mAP_dictionary is not None:
        current_mAP = get_current_mAP(current_embeddings = visual_embeddings, **mAP_dictionary)
        historical_mAP.append(current_mAP)
    else:
        historical_mAP.append(0)

    for op in range(iterations + 1):

        print("Training iteration", op)

        with tf.GradientTape() as t:
            new_visual_embeddings = gnn(visual_embeddings, similarity_text, lyr, tf.nn.relu)

            similarity_visual = similarity_func(new_visual_embeddings)

            if adjust_range:
                curr_similarity_text, curr_similarity_visual = reduce_range_to_0_to_1(similarity_text, similarity_visual)
            else:
                curr_similarity_text = similarity_text
                curr_similarity_visual = similarity_visual

            batch_mask = get_k_random_pairs(similarity=curr_similarity_text.numpy(), k=100)

            loss = loss_by_visual_text_contrast(curr_similarity_visual, curr_similarity_text, batch_mask)

        variables = t.watched_variables()
        grads = t.gradient(loss, variables)

        optimizer.apply_gradients(zip(grads, variables))

        historical_loss.append(loss.numpy().item())

        if mAP_dictionary is not None:
            window = int(iterations/historical_mAP_maxsize)
            if (op % window) == 0:
                current_mAP = get_current_mAP(current_embeddings = new_visual_embeddings.numpy(), **mAP_dictionary)
                historical_mAP.append(current_mAP)
        else:
            historical_mAP.append(0)

    return new_visual_embeddings.numpy(), historical_loss, historical_mAP


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)                
    parser.add_argument("-mode", type=str, choices=['compute', 'utils', "gnn"], help=" mode of operation", required=True)
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

    if pargs.mode == 'utils':
        ssearch.load_features()
        prepare_dataset(dataset, ssearch.features, ssearch.filenames, use_umap=True, n_components=128)   

    if pargs.mode == 'gnn':

        # Text embeddings model
        model_name = "RoBERTa"
        
        # Quantity of training iterations
        iterations = 30

        #test_set = "catalogue"
        test_set = "test_folder"

        train_visuals = True
        adjust_query = True

        visual_embeddings = np.load("./catalogues/{}/embeddings/ResNet/visual_embeddings.npy".format(dataset))
        text_embeddings = np.load("./catalogues/{}/embeddings/{}/text_embeddings.npy".format(dataset, model_name))
        
        visual_embeddings_catalog = "./catalogues/{}/ssearch/catalog.txt".format(dataset)
        text_embeddings_catalog = "./catalogues/{}/ssearch/text_embeddings_catalog.txt".format(dataset)
        text_embeddings = reorder_text_embeddings(text_embeddings, visual_embeddings_catalog, text_embeddings_catalog)

        ssearch.features = visual_embeddings
        ssearch.enable_search = True
        original_features = np.copy(ssearch.features)

        metric = 'cos'
        norm = 'None'

        data_path = "./catalogues/{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        real_df = None
        eval_path = "./catalogues/{}/test".format(dataset)
        eval_files = ["./catalogues/{}/test/".format(dataset) + f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

        mAP_dictionary = {"ssearch" : ssearch,
                          "eval_files" : eval_files,
                          "metric" : metric,
                          "norm" : norm,
                          "adjust_query" : adjust_query,
                          "original_embeddings" : original_features,
                          "df" : df,
                          "real_df" : real_df}

        if not train_visuals:
            new_visual_embeddings = visual_embeddings

        else:
            new_visual_embeddings, historical_loss, historical_mAP = train_visual(visual_embeddings, text_embeddings, iterations, 0.01, mAP_dictionary)
            plot_loss_and_mAP(historical_loss, historical_mAP)

        ssearch.features = new_visual_embeddings

        if pargs.list is None:
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []
            
            if test_set == "test_folder":

                save_results = False

                for fquery in eval_files:
                    im_query = ssearch.read_image(fquery)
                    idx, dist_sorted, q_fv, data_search = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_features, df=df)         
                    r_filenames = ssearch.get_filenames(idx)
                    
                    r_filenames.insert(0, fquery)
                    base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                    ap, ap_tree, ap_sub = avg_precision(base_category, products)
                    ap_arr.append(ap)
                    ap_arr_tree.append(ap_tree)
                    ap_arr_sub.append(ap_sub)

                    if save_results:
                        image_r= ssearch.draw_result(r_filenames)
                        output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm)
                        output_name = os.path.join("./catalogues/{}/results".format(dataset), output_name)
                        io.imsave(output_name, image_r)
                        print('result saved at {}'.format(output_name))

            elif test_set == "catalogue":

                for im_query_idx in range(original_features.shape[0]):

                    if im_query_idx % 10 == 0:
                        print("Catalogue item", im_query_idx)

                    fquery = ssearch.filenames[im_query_idx]

                    im_query = original_features[im_query_idx].reshape(1, -1)
                    idx, dist_sorted, q_fv, data_search = ssearch.search(im_query, metric=metric, norm=norm, top=20, adjust_query=adjust_query, original_embeddings=original_features, df=df, query_is_feature=True)         
                    r_filenames = ssearch.get_filenames(idx)
                    
                    r_filenames.insert(0, fquery)
                    base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
                    ap, ap_tree, ap_sub = avg_precision(base_category, products)
                    ap_arr.append(ap)
                    ap_arr_tree.append(ap_tree)
                    ap_arr_sub.append(ap_sub)

            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP(GC): {}, mAP(CT): {}".format(mAP, mAP_tree))