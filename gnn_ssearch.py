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

import json

from similarity import *
from loss import *
from save_sims import *
from plot_utils import *


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
        catalog_file = os.path.join(self.ssearch_dir, 'visual_embeddings_catalog.txt')        
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

    def adjust_query_embedding_old(self, query, original_embeddings, top=3, decide=True, df=None):
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


    def adjust_query_embedding(self, query, original_embeddings, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(visual_embeddings[0]))

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
            q_fv = self.adjust_query_embedding(query=q_fv, original_embeddings=original_embeddings, top=3, df=df, decide=False)
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


def get_current_mAP_og(current_embeddings,
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


def get_current_mAP(current_embeddings,
                    current_iteration,
                    sim_visual,
                    ssearch,
                    visual_embeddings_test,
                    test_indexes,
                    eval_files,
                    metric,
                    norm,
                    adjust_query,
                    original_embeddings,
                    df,
                    real_df,
                    top):

    ssearch.features = current_embeddings
    
    ap_arr = []
    ap_arr_tree = []
    ap_arr_sub = []

    save_labels = True
    save_images = False

    print("Computing mAP")

    results_dict = {
        "idx_dict" : {}
    }

    GC_list_all = []
    CT_list_all = []
    SC_list_all = []

    for i in range(len(eval_files)):
        fquery = eval_files[i]

        if test_indexes is not None:
            # The query is taken from the training set
            im_query = current_embeddings[test_indexes[i]].reshape(1, -1)
        else:
            # The query does not come from the training set,
            # its visual embeddings have been computed before training
            im_query = visual_embeddings_test[i].reshape(1, -1)

        idx, dist_sorted, q_fv, data_search = ssearch.search(im_query, metric=metric, norm=norm, top=top, adjust_query=adjust_query, original_embeddings=original_embeddings, df=df, query_is_feature=True)         
        r_filenames = ssearch.get_filenames(idx)
        
        base_category, products = get_product_and_category(r_filenames, dataframe=df, real_df=real_df)
        ap, ap_tree, ap_sub = avg_precision(base_category, products)
        ap_arr.append(ap)
        ap_arr_tree.append(ap_tree)
        ap_arr_sub.append(ap_sub)

        if save_labels:

            GC_list = [base_category[0]]
            CT_list = [base_category[1]]
            SC_list = [base_category[2]]

            for product in products:
                GC_list.append(product[1])
                CT_list.append(product[2])
                SC_list.append(product[3])
            
            GC_list_all.append(GC_list)
            CT_list_all.append(CT_list)
            SC_list_all.append(SC_list)

        if test_indexes is not None:
            results_dict["idx_dict"][test_indexes[i].item()] = []

            for result_idx in idx:
                similarity_score = sim_visual[test_indexes[i], result_idx].numpy()
                results_dict["idx_dict"][test_indexes[i].item()].append([result_idx.item(), similarity_score.item()])

        if save_images:
            image_r= ssearch.draw_result(r_filenames)
            output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm)
            output_name = os.path.join("./catalogues/{}/results/iter_{}".format(dataset, current_iteration), output_name)
            io.imsave(output_name, image_r)
            print('result saved at {}'.format(output_name))

    mAP = statistics.mean(ap_arr)
    mAP_tree = statistics.mean(ap_arr_tree)
    mAP_sub = statistics.mean(ap_arr_sub)
    print("mAP(GC): {}, mAP(CT): {}, mAP(SC): {}".format(mAP, mAP_tree, mAP_sub))

    if save_labels:
        save_result_labels_csv(GC_list_all, CT_list_all, SC_list_all, dataset, current_iteration)

    results_dict["mAP"] = mAP

    return mAP, results_dict


def reorder_text_embeddings(text_embeddings, ve_catalog, te_catalog):
    # Necessary for both the visual and text embeddings
    # to be in the same order

    with open(ve_catalog, 'r') as file_ve, open(te_catalog, 'r') as file_te:
        lines_ve = file_ve.read().splitlines()
        lines_te = file_te.read().splitlines()

    indices = np.zeros(len(lines_ve), dtype=np.int32)

    for i, element in enumerate(lines_ve):
        indices[i] = lines_te.index(element[0:-4])

    reordered_text_embeddings = text_embeddings[indices]
    return reordered_text_embeddings


def get_k_random_pairs(similarity, k = 50, alpha = 10):

    # 2D indexes for the similarity matrix:
    similarity_idx = np.triu_indices(similarity.shape[0])

    # 1D array with similarity scores of upper half of matrix
    similarity_triu = similarity[similarity_idx]

    #plot_histogram_sim(similarity_triu)

    # 1D indexes for the 2D indexes for the similarity matrix:
    indexes = np.arange(similarity_idx[0].shape[0])

    probs = 1 * similarity_triu
    probs -= np.min(probs)
    probs /= np.max(probs)
    probs *= 2 * alpha
    probs -= alpha
    probs = np.exp(probs)

    probs -= np.min(probs)
    probs /= np.sum(probs)
    random_indexes_similar = np.random.choice(indexes, k, replace=False, p=probs)

    #print("sim_min", np.min(probs), "sim_sum", np.sum(probs))
    #plot_probs(similarity_triu, probs)

    probs = -1 * similarity_triu
    probs -= np.min(probs)
    probs /= np.max(probs)
    probs *= 2 * alpha
    probs -= alpha
    probs = np.exp(probs)

    probs -= np.min(probs)
    probs /= np.sum(probs)
    random_indexes_dissimilar = np.random.choice(indexes, k, replace=False, p=probs)

    #print("dissim_min", np.min(probs), "dissim_sum", np.sum(probs))
    #plot_probs(similarity_triu, probs)

    #plot_histogram_randoms(similarity_triu, random_indexes_similar, random_indexes_dissimilar)
    
    batch_indexes = np.concatenate((random_indexes_similar, random_indexes_dissimilar))

    return [similarity_idx[0][batch_indexes], similarity_idx[1][batch_indexes]]


def reduce_range_to_0_to_1(sim_text, sim_visual, margin=0.01):

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


def gnn(fts, adj, transform, activation):
    seq_fts = transform(fts)
    ret_fts = tf.matmul(adj, seq_fts)
    return activation(ret_fts)


def train_visual(visual_embeddings, text_embeddings, iterations, lr, mAP_dictionary = None, test_w_train_set = False):
    #mAP_dictionary = None

    adjust_range = True

    similarity_func = get_cos_similarity_tensor
    #similarity_func = get_cos_softmax_similarity_tensor
    #similarity_func = get_sqrt_similarity_tensor
    #similarity_func = get_sqrt_normmin_similarity_tensor

    similarity_text = similarity_func(text_embeddings)
    similarity_visual = similarity_func(visual_embeddings)

    #save_most_similar_pairs(similarity_text.numpy(), dataset)

    #units = 1024
    units = visual_embeddings.shape[1]
    lyr = tf.keras.layers.Dense(units)

    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # For plotting purposes:
    historical_loss = {"iters" : [], "losses" : []}
    historical_mAP = {"iters" : [], "mAPs" : []}

    # Similarity, prob_gt and prob_pred values by iteration to be saved to an .xlsx file
    all_sims_and_probs = {}

    # Search results by iteration to be viewed via web interface
    search_results_list = {}

    if mAP_dictionary is not None:

        # OJO: CURRENT_EMBEDDINGS DEBE SER VISUAL, NO TEXTUAL
        current_mAP, results_dict = get_current_mAP(current_embeddings = visual_embeddings,
                                                    current_iteration = 0,
                                                    sim_visual = similarity_visual,
                                                    **mAP_dictionary)
        search_results_list[0] = results_dict
        
        historical_mAP["iters"].append(0)
        historical_mAP["mAPs"].append(current_mAP)

    for it in range(1, iterations+1):

        print("Training iteration", it)

        with tf.GradientTape() as t:
            new_visual_embeddings = gnn(visual_embeddings, similarity_text, lyr, tf.nn.relu)

            similarity_visual = similarity_func(new_visual_embeddings)

            if adjust_range:
                curr_similarity_text, similarity_visual = reduce_range_to_0_to_1(similarity_text, similarity_visual)
            else:
                curr_similarity_text = similarity_text

            batch_indexes = get_k_random_pairs(similarity=curr_similarity_text.numpy(), k=100)

            #loss = loss_by_visual_text_contrast(similarity_visual, curr_similarity_text, batch_indexes)
            loss, sims_and_probs = loss_unet(similarity_visual, curr_similarity_text, batch_indexes)

        variables = t.watched_variables()
        grads = t.gradient(loss, variables)

        optimizer.apply_gradients(zip(grads, variables))

        # Saving various results:

        historical_loss["iters"].append(it)
        historical_loss["losses"].append(loss.numpy().item())

        eval_window = 10

        if (it % eval_window) == 1:

            all_sims_and_probs["iter_" + str(it)] = sims_and_probs

            if mAP_dictionary is not None:

                current_mAP, results_dict = get_current_mAP(current_embeddings = new_visual_embeddings.numpy(),
                                                            current_iteration = it,
                                                            sim_visual = similarity_visual,
                                                            **mAP_dictionary)
                search_results_list[it] = results_dict

                historical_mAP["iters"].append(it)
                historical_mAP["mAPs"].append(current_mAP)

    plot_loss(historical_loss)
    plot_mAP(historical_mAP)
    #plot_prob_gt(all_sims_and_probs)
    
    save_sims_and_probs(all_sims_and_probs, dataset)
    
    if test_w_train_set:
        # JSON to be used by the "similar_explorer" web app
        with open("./similar_explorer/similars.json", "w") as json_file:
            json.dump(search_results_list, json_file)
    
    return new_visual_embeddings.numpy()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)                
    parser.add_argument("-mode", type=str, choices=['compute', 'compute_test_queries', 'utils', "gnn"], help=" mode of operation", required=True)
    parser.add_argument('-umap', action='store_true')
    parser.add_argument('-real', action='store_true', help="whether to use real images or not when evaluating")
    parser.add_argument("-dataset",  type=str, choices=['Pepeganga', 'PepegangaCLIPBASE', 'Cartier', 'CartierCLIPBASE', 'IKEA', 'IKEACLIPBASE', 'UNIQLO', 'UNIQLOCLIPBASE', 'WorldMarket', 'WorldMarketCLIPBASE', 'Homy', 'HomyCLIPBASE'], help="dataset", required=True)
    parser.add_argument("-list", type=str,  help=" list of image to process", required=False)
    parser.add_argument("-odir", type=str,  help=" output dir", required=False, default='.')
    pargs = parser.parse_args()
    configuration_file = pargs.config
    ssearch = SSearch(pargs.config, pargs.name)
    norm = 'None'
    
    dataset = pargs.dataset
    use_real_queries = pargs.real

    if pargs.mode == 'compute':        
        ssearch.compute_features_from_catalog()
    
    if pargs.mode == 'compute_test_queries':

        # Prepares files for eval files, for testing with items both inside and outside
        # of the training set

        eval_path = "./catalogues/{}/test".format(dataset)
        eval_files = [f for f in os.listdir(eval_path) if os.path.isfile(join(eval_path, f))]

        # The following 2 lines are run to get the dimension of the generated queries
        im_query = ssearch.read_image("./catalogues/{}/test/".format(dataset) + eval_files[0])
        q_fv = ssearch.compute_features(im_query, expand_dims = True)
        # ----------------

        test_fv = np.empty((len(eval_files), q_fv.shape[1]), dtype = np.float32)

        filename_ve_test = "./catalogues/{}/ssearch/visual_embeddings_test_catalog.txt".format(dataset)

        # Save text file with the filenames of all image queries
        with open(filename_ve_test, 'w') as file_ve_test:

            for i, fquery in enumerate(eval_files):

                file_ve_test.write(fquery + '\n')

                fquery_full_path = "./catalogues/{}/test/".format(dataset) + fquery
                im_query = ssearch.read_image(fquery_full_path)
                q_fv = ssearch.compute_features(im_query, expand_dims = True)
                test_fv[i, ] = q_fv
        
        # Save numpy backup with the visual embeddings of all queries
        np.save("./catalogues/{}/embeddings/ResNet/visual_embeddings_test.npy".format(dataset), test_fv)

        # The following lines are to get the size of the training dataset
        filename_ve = "./catalogues/{}/ssearch/visual_embeddings_catalog.txt".format(dataset)

        with open(filename_ve, 'r') as file_ve:
            lines_ve = file_ve.read().splitlines()
        # ----------------

        random_integers = np.random.randint(0, len(lines_ve), len(eval_files))
        random_integers = np.sort(random_integers)

        # Save numpy backup with indexes of the training dataset to be used for testing
        np.save("./catalogues/{}/ssearch/test_set_integers.npy".format(dataset), random_integers)

        # Save JSON with indexes of the training dataset to be used by the "similar_explorer" web app
        with open('./similar_explorer/test_set_integers.json', 'w') as json_file:
            json.dump(random_integers.tolist(), json_file)


    if pargs.mode == 'utils':
        ssearch.load_features()
        prepare_dataset(dataset, ssearch.features, ssearch.filenames, use_umap=True, n_components=128)   

    if pargs.mode == 'gnn':

        # Text embeddings model.
        model_name = "RoBERTa"
        
        # Quantity of training iterations.
        iterations = 101

        # Whether to use in-training visual embeddings as queries.
        # Remember to run gnn_search.py with -mode compute_test_queries before.
        test_w_train_set = False

        # Whether to use VETE-B query adjustment or not.
        adjust_query = False

        # Quantity of results to get in search.
        results_per_query = 20

        visual_embeddings = np.load("./catalogues/{}/embeddings/ResNet/visual_embeddings.npy".format(dataset))
        text_embeddings = np.load("./catalogues/{}/embeddings/{}/text_embeddings.npy".format(dataset, model_name))
        
        visual_embeddings_catalog = "./catalogues/{}/ssearch/visual_embeddings_catalog.txt".format(dataset)
        text_embeddings_catalog = "./catalogues/{}/ssearch/text_embeddings_catalog.txt".format(dataset)
        text_embeddings = reorder_text_embeddings(text_embeddings, visual_embeddings_catalog, text_embeddings_catalog)

        ssearch.features = visual_embeddings
        ssearch.enable_search = True
        original_features = np.copy(ssearch.features)

        metric = 'cos'
        #metric = "l2"
        norm = 'None'

        data_path = "./catalogues/{}/data/".format(dataset)
        df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

        real_df = None

        if test_w_train_set:

            visual_embeddings_test = None

            with open(visual_embeddings_catalog, 'r') as file_ve:
                lines_ve = file_ve.read().splitlines()

            test_indexes = np.load("./catalogues/{}/ssearch/test_set_integers.npy".format(dataset))
            eval_files = ["./catalogues/{}/test_from_train_set/".format(dataset) + lines_ve[idx] for idx in test_indexes]

        else:

            visual_embeddings_test = np.load("./catalogues/{}/embeddings/ResNet/visual_embeddings_test.npy".format(dataset))
            visual_embeddings_test_catalog = "./catalogues/{}/ssearch/visual_embeddings_test_catalog.txt".format(dataset)

            with open(visual_embeddings_test_catalog, 'r') as file_ve:
                lines_ve = file_ve.read().splitlines()
            
            test_indexes = None
            eval_files = ["./catalogues/{}/test/".format(dataset) + line for line in lines_ve]


        mAP_dictionary = {"ssearch" : ssearch,
                          "visual_embeddings_test": visual_embeddings_test,
                          "test_indexes": test_indexes,
                          "eval_files" : eval_files,
                          "metric" : metric,
                          "norm" : norm,
                          "adjust_query" : adjust_query,
                          "original_embeddings" : original_features,
                          "df" : df,
                          "real_df" : real_df,
                          "top" : results_per_query}
        

        # lr = 0.01
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03,
                                                                  decay_steps = iterations)
        
        new_visual_embeddings = train_visual(visual_embeddings, text_embeddings, iterations, lr_decayed_fn, mAP_dictionary, test_w_train_set)
        


"""         ssearch.features = new_visual_embeddings

        if pargs.list is None:
            ap_arr = []
            ap_arr_tree = []
            ap_arr_sub = []

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

            mAP = statistics.mean(ap_arr)
            mAP_tree = statistics.mean(ap_arr_tree)
            mAP_sub = statistics.mean(ap_arr_sub)
            print("mAP(GC): {}, mAP(CT): {}".format(mAP, mAP_tree)) """