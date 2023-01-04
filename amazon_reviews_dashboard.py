import os
import time
import json

import os.path as osp
import PySimpleGUI as sg

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

def is_valid_path(filepath):
    if filepath and osp(filepath).exists():
        return True
    sg.popup_error("Filepath not correct")
    return False

def load_data_window():
    try:
        sg.theme('DarkGray')
        load_layout = [[sg.Text("Dataset:", s=15, justification="r"), sg.I(key="-IN-"), sg.FolderBrowse()],
                        [sg.Button('Load Data')]]

        load_win = sg.Window("Load Data", load_layout)

        filename_switcher = {'Music': ['CDs_and_Vinyl_5', 'meta_CDs_and_Vinyl', 'Music_embeds'],
                            'Kitchen': ['Prime_Pantry_5', 'meta_Prime_Pantry', 'Kitchen_embeds'],
                            'Pets': ['Pet_Supplies_5', 'meta_Pet_Supplies', 'Pets_embeds']}

        while True:
            event, values = load_win.read()
            if event in (sg.WINDOW_CLOSED, "Exit"):
                break
            elif event == '-IN-':
                print(values["-IN-"])
            elif event == 'Load Data':
                products_type = osp.basename(osp.normpath(values["-IN-"]))

                if products_type not in filename_switcher:
                    sg.popup('Dataset Load Error.\n I am configured to load data from folders named Music, Kitchen, or Pets.\n\n Please try to find one of these folders.')
                    break                    
                
                data_file = osp.join(values["-IN-"], filename_switcher[products_type][0] + '.json')
                meta_file = osp.join(values["-IN-"], filename_switcher[products_type][1] + '.json')
                try:
                    embed_sort_per_asin = {}
                    embed_sort_file = osp.join(values["-IN-"], filename_switcher[products_type][2] + '.json')
                    embed_sort_data_in_raw = read_json(embed_sort_file)
                    embed_sort_per_asin = sort_reviews_by_product(embed_sort_data_in_raw)
                    print(f'Loaded Embedding file from {embed_sort_file}')
                except:
                    pass
                print(f'data_file {data_file}')

                if not osp.exists(data_file):
                    print('inside data exist')
                    sg.popup(title='Dataset Load Error:')
                    print(f'data_file {data_file}')
                
                if not osp.exists(meta_file):
                    sg.popup(f'Metadata Load Error. Unable to Find metadata at: {meta_file}')
                    print(f'meta_file {meta_file}')

                try:
                    reviews_struct = read_json(data_file)
                    meta_struct = read_json(meta_file)
                except Exception as e0:
                    reviews_struct = {}
                    meta_struct = {}
                
                load_win.close()
                return values["-IN-"], reviews_struct, meta_struct, embed_sort_per_asin
        load_win.close()
    except Exception as e:
        sg.popup_error_with_traceback(f'Unable to Load data via load_data_window.  Here is the info:', e)

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def drawChart(window, fig):
    # fig.get_tk_widget().forget()
    # plt.clf()
    fig_agg = draw_figure(
        window['figCanvas'].TKCanvas, fig)
    return fig_agg

# def updateChart(fig_agg, window, fig):
#     fig_agg.get_tk_widget().forget()
#     # plt.cla()
#     plt.clf()
#     fig_agg = draw_figure(
#         window['figCanvas'].TKCanvas, fig)

def get_product_title_from_meta(asin_in, atitle_list):
    product_title = ''
    try:
        for atl in atitle_list:
            if asin_in == atl[0]:
                return atl[1]
    except Exception as e:
        print(f'{e}')

    return product_title

def read_json(filepath):
    with open(filepath,'r') as f:
        texts = f.readlines()

    json_out = [json.loads(txt) for txt in texts]
    return json_out

def sort_reviews_by_product(full_reviews):
    reviews_per_product = {}
    try:
        for fr in tqdm(full_reviews):
            if fr:
                asin_tmp = fr['asin']
                if asin_tmp in reviews_per_product:
                    reviews_per_product[asin_tmp].append(fr)
                else:
                    reviews_per_product[asin_tmp] = []
                    reviews_per_product[asin_tmp].append(fr)
    except Exception as e:
        print(f'Unable to Sort Reviews by Product: {e}')
    return reviews_per_product

def run_basic_analysis(full_reviews, meta_data_in_raw,**kwargs):

    product_group = kwargs.get('product_group', 'Kitchen')

    # How many individual Products?
    all_products = [fr['asin'] for fr in full_reviews]
    unique_products = np.unique(all_products)

    status_text = f'Successfully Loaded {product_group} Dataset: There are {len(full_reviews)} reviews and {len(unique_products)} Unique products'
    print(status_text)

    reviews_per_product = sort_reviews_by_product(full_reviews)
    
    # status_text = f'Sorted Reviews per Item, {len(list(reviews_per_product.keys()))} Items Collated'
    # print(status_text)

    dict_key = 'overall'
    item_list = list(reviews_per_product.keys())

    overall_score_dict = {}
    for product in item_list:
        key_list_formap = [dict_key] * len(reviews_per_product[product])
        overall_scores = map(get_field, reviews_per_product[product], key_list_formap)
        overall_score_dict[product] = np.array(list(overall_scores), dtype=int)
    
    avg = np.array([np.mean(value) for key, value in overall_score_dict.items()])
    count = np.array([np.size(value) for key, value in overall_score_dict.items()])

    return status_text, overall_score_dict, avg, count

def get_product_title_from_meta(asin_in, atitle_list):
    product_title = ''
    try:
        for atl in atitle_list:
            if asin_in == atl[0]:
                return atl[1]
    except Exception as e:
        print(f'{e}')

    return product_title

def sort_ratings(overall_score_dict, asin_to_title):
    sorted_score_freq_dict = {k: len(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: len(overall_score_dict[k]), reverse=True) }
    sorted_score_mean_dict = {k: np.mean(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.mean(overall_score_dict[k]), reverse=True) }
    sorted_score_std_dict = {k: np.std(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.std(overall_score_dict[k]), reverse=True) }
    sorted_score_med_dict = {k: np.median(overall_score_dict[k]) for k in sorted(overall_score_dict, key=lambda k: np.median(overall_score_dict[k]), reverse=True) }

    sorted_ids_freq = list(sorted_score_freq_dict.keys())
    sorted_titles_freq = [get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_freq]
    # print(f'Most reviewed items: {sorted_titles_freq[:10]}')
    # print(f'Least reviewed Items: {[get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_freq[-10:]]}\n')

    sorted_ids_mean = list(sorted_score_mean_dict.keys())
    sorted_titles_mean = [get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_mean]
    # print(f'Highest Average Ratings: {sorted_titles_mean[:10]}')
    # print(f'Lowest Average Ratings: {sorted_titles_mean[-10:]}\n')

    sorted_ids_std = list(sorted_score_std_dict.keys())
    sorted_titles_std = [get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_std]
    # print(f'Least Consistent Ratings (standard deviation): {sorted_titles_std[:10]}')
    # print(f'Most Consistent Ratings (standard deviation): {sorted_titles_std[-10:]}\n')

    sorted_ids_med = list(sorted_score_med_dict.keys())
    sorted_titles_med = [get_product_title_from_meta(sf, asin_to_title) for sf in sorted_ids_med]
    # print(f'Highest Median Ratings: {sorted_titles_med[:10]}')
    # print(f'Lowest Median Ratings: {sorted_titles_med[-10:]}\n')

    return sorted_titles_freq, sorted_titles_mean, sorted_titles_std, sorted_titles_med

def sort_helpful_reviews(reviews_per_product):
    sorted_recommended_reviews = {}
    try:
        # get recommended reviews
        recommended_reviews_per_item = {}
        for key, review_per_product in tqdm(reviews_per_product.items()):
            tmp_list = []
            for fr in review_per_product:
                if 'vote' in list(fr.keys()):
                    # print(fr['vote'])
                    try:
                        tmp_list.append(fr)
                    except:
                        pass
            recommended_reviews_per_item[key] = tmp_list

        # Extract Number of helpful votes per review
        num_votes_dict = {}
        for product in list(recommended_reviews_per_item.keys()):
            key_list_formap = ['vote'] * len(recommended_reviews_per_item[product])
            overall_scores = map(get_field, recommended_reviews_per_item[product], key_list_formap)
            num_votes_dict[product] = np.array(list(overall_scores), dtype=int)

        # sort the reviews per item based on votes per review
        for product, review_list in recommended_reviews_per_item.items():
            if len(review_list) == 0:
                continue
            sorted_recommended_reviews[product] = [review_list[i] for i in np.argsort(num_votes_dict[product])][::-1]

    except Exception as e:
        print(f'Unable to Sort Recommended Reviews: {e}')
    return sorted_recommended_reviews

def get_field(input_dict, field):
    return input_dict[field]

def scatter_plot_ratings(avg, count, product_type):
    # Plot Scores
    fig, ax = plt.subplots(1, 1)
    ax.scatter(count,avg, c='k')
    ax.set_xlabel('Number of Reviews')
    ax.set_ylabel('Average Rating')
    ax.set_title(f'Distribution of Reviews per Product\n {product_type} Dataset')

    # plt.show()
    return fig

def get_title_from_asin(asin_to_title, asin, **kwargs):
    
    get_title = kwargs.get('get_title', True)
    title = ''

    try:
        if get_title:
            for a in asin_to_title:
                if a[0] == asin:
                    return a[1]
        else:
            for a in asin_to_title:
                if a[1] == asin:
                    return a[0]
    except:
        print('Failed get_title_from_asin')
    return title

## Main
def main():
    sg.theme('DarkGray')

    menu_def = [["File", ['Load Data', 'Exit']],
                ["Help", ["Settings", "About"]]]

    layout = [[sg.MenubarCustom(menu_def, tearoff=False)],
              [sg.Text(text="Please begin by Loading data via File->Load Data", key='status_text'),
              sg.Text("Most Helpful Reviews of Selected Product", pad=(225, 0), key='review-reader-status'),
              sg.Text("Exemplar Reviews of Selected Product", pad=(65, 0), key='review-reader-status')],
              [sg.Canvas(key='figCanvas'), sg.Listbox(["Selected Reviews Will Display Here"], size=(60,5), pad=(0, 0), enable_events=False, key='review-reader-0'),
              sg.Listbox(["Selected Exemplar Reviews Will Display Here"], size=(60,5), pad=(0, 0), enable_events=False, key='review-reader-1'),],
              [sg.Text(text="Please Choose an Evaluation Criterion", key='eval_crit_text'), sg.Text(text="Highest Rated Products", pad=(200, 0), key='high_drop_text'),
              sg.Text(text="Lowest Rated Products", pad=(25, 0), key='low_drop_text')],
              [sg.Listbox(['Number of Reviews', 'Avg Rating', 'Rating St Dev', 'Rating Median'], size=(18,5), enable_events=True, key='rating-drop-0'),
              sg.Listbox(["Highest Rated Products"], size=(60,5), enable_events=True, key='rating-drop-1'),
              sg.Listbox(["Lowest Rated Products"], size=(60,5), enable_events=True, key='rating-drop-2')],
              [sg.Exit()]]
    
    product_types = ['Music', 'Kitchen', 'Pets']
    dat_files = ['CDs_and_Vinyl_5', 'Prime_Pantry_5', 'Pet_Supplies_5']
    meta_files = ['meta_CDs_and_Vinyl', 'meta_Prime_Pantry', 'meta_Pet_Supplies']
    product_type_ind = 0

    filename_switcher = {'Music': ['CDs_and_Vinyl_5', 'meta_CDs_and_Vinyl'],
                        'Kitchen': ['Prime_Pantry_5', 'meta_Prime_Pantry', 'embed_sort_Prime_Pantry'],
                        'Pets': ['Pet_Supplies_5', 'meta_Pet_Supplies']}
    
    # Create the window
    window = sg.Window("Amazon Reviews Dashboard", layout,
                        use_custom_titlebar=True,
                        resizable=True,
                        finalize=True)
    
    # Try to load the data
    try:
        
        data_path = 'D:\\Coding\\sandbox\\Amzn_Stats\\Amzn_data\\' + product_types[1]
        # data_path = osp.join(os.getcwd(), 'Amzn_data', product_types[1])
        print(f'data dir = {data_path}')

        if osp.exists(data_path):
            status_text = f'Attempting to load dataset from {data_path}'
            event, values = window.read(timeout=1)
            window['status_text'].update(status_text)
            time.sleep(1)

        try:
            # filenames
            products_type = osp.basename(osp.normpath(data_path))
            data_file = osp.join(data_path, filename_switcher[products_type][0] + '.json')
            meta_file = osp.join(data_path, filename_switcher[products_type][1] + '.json')
            try:
                embed_sort_per_asin = {}
                embed_sort_file = osp.join(data_path, filename_switcher[products_type][2] + '.json')
                embed_sort_data_in_raw = read_json(embed_sort_file)
                embed_sort_per_asin = sort_reviews_by_product(embed_sort_data_in_raw)
                print(f'Loaded Embedding file from {embed_sort_file}')
            except:
                pass

            # import 
            full_reviews = read_json(data_file)
            meta_data_in_raw = read_json(meta_file)

        except Exception as e:
            data_path, full_reviews, meta_data_in_raw, embed_sort_data_in_raw = load_data_window()
        

    except Exception as e:
        sg.popup_error_with_traceback(f'Load error.  Here is the info:', e)

    # Run Initial Analysis Automatically
    status_text, overall_score_dict, avg, count = run_basic_analysis(full_reviews, meta_data_in_raw, product_group=products_type)
    window['status_text'].update(status_text)
    product_group = products_type
    figure_handle = scatter_plot_ratings(avg, count, products_type)
    drawChart(window, figure_handle, )
    

    # item_list = list(reviews_per_product.keys())

    # Best and Worst Dropdowns
    asin_to_title = [[m['asin'], m['title']]for m in meta_data_in_raw]
    sorted_titles_freq, sorted_titles_mean, sorted_titles_std, sorted_titles_med  = sort_ratings(overall_score_dict, asin_to_title)

    # Recommended Reviews
    reviews_per_product = sort_reviews_by_product(full_reviews)
    sorted_recommended_reviews = sort_helpful_reviews(reviews_per_product)

    try:
        # UI Event loop
        while True:
            event, values = window.read(timeout=1)
            # End program if user closes window
            if event in (None, sg.WIN_CLOSED, 'Exit'):
                break
            elif event == 'Load Data':
                try:
                    data_path, full_reviews, meta_data_in_raw,embed_sort_data_in_raw = load_data_window()
                    product_group = osp.basename(osp.normpath(data_path))
                    status_text, overall_score_dict, avg, count = run_basic_analysis(full_reviews, meta_data_in_raw, product_group=product_group)
                    window['status_text'].update(status_text)
                    figure_handle = scatter_plot_ratings(avg, count, product_group)
                    drawChart(window, figure_handle)

                    # Best and Worst Dropdowns
                    asin_to_title = [[m['asin'], m['title']]for m in meta_data_in_raw]
                    sorted_titles_freq, sorted_titles_mean, sorted_titles_std, sorted_titles_med = sort_ratings(overall_score_dict, asin_to_title)

                    # Recommended Reviews
                    reviews_per_product = sort_reviews_by_product(full_reviews)
                    sorted_recommended_reviews = sort_helpful_reviews(reviews_per_product)
                except Exception as e:
                    print('Failed to Load New Dataset')

            elif event == 'rating-drop-0':
                switcher = {'Number of Reviews': sorted_titles_freq,
                            'Avg Rating': sorted_titles_mean,
                            'Rating St Dev': sorted_titles_std,
                            'Rating Median': sorted_titles_med}
                title_switcher = {'Number of Reviews': ['Most Reviewed Products', 'Least Reviewed Products'],
                                  'Avg Rating': ['Highest Rated Products', 'Lowest Rated Products'],
                                  'Rating St Dev': ['Least Consistently Rated Products', 'Most Consistently Rated Products'],
                                  'Rating Median': ['Highest Median Ratings', 'Lowest Median Ratings']}

                tmp_str = values['rating-drop-0'][0]

                # set dropdowns
                window['rating-drop-1'].update(switcher[tmp_str][:10])
                window['rating-drop-2'].update(switcher[tmp_str][-10:])
                window['high_drop_text'].update(title_switcher[tmp_str][0])
                window['low_drop_text'].update(title_switcher[tmp_str][1])

                # Update Figure
                # if tmp_str == 'Rating St Dev':
                #     std = np.array([np.std(value) for key, value in overall_score_dict.items()])
                #     figure_handle = scatter_plot_ratings(std, count, product_group)
                #     update_chart(window, figure_handle)

            elif event == 'rating-drop-1':
                # get recommended review text for selected product
                rev_exists = True
                num_to_print = 3

                try:
                    tmp_asin = get_title_from_asin(asin_to_title, values['rating-drop-1'][0], get_title=False)
                    selected_reviews = sorted_recommended_reviews[tmp_asin]
                except Exception as e:
                    rev_exists = False
                    product_title = values['rating-drop-1'][0]
                    review_text_to_print = [f'No recommended reviews for: {product_title}']
                    print(f'No recommended reviews for: {product_title}')

                if rev_exists:
                    if len(selected_reviews) < num_to_print:
                        review_text_to_print = [r['reviewText'] for r in selected_reviews]
                    else:
                        review_text_to_print = [r['reviewText'] for r in selected_reviews[:num_to_print]]

                # print(f'sorted_recommended_reviews: {review_text_to_print} \n\n')
                # window['review-reader-0'].update(review_text_to_print)

                    
                window['review-reader-0'].update(review_text_to_print)

                # Show Embedding Analysis
                exemplar_exists = True
                try:
                    exemplar_reviews = embed_sort_per_asin[tmp_asin]
                    # print(f'Exemplars {exemplar_reviews[:3]}')
                except Exception as e:
                    exemplar_text_to_print = [f'No exemplar reviews for: {product_title}']
                    print(f'No recommended reviews for: {exemplar_text_to_print}')
                
                if exemplar_exists:
                    if len(exemplar_reviews) < num_to_print:
                        exemplar_text_to_print = [r['reviewText'] for r in exemplar_reviews]
                    else:
                        exemplar_text_to_print = [r['reviewText'] for r in exemplar_reviews[:num_to_print]]
                
                window['review-reader-1'].update(exemplar_text_to_print)

            elif event == 'rating-drop-2':
                # get recommended review text for selected product
                rev_exists = True
                num_to_print = 3

                try:
                    tmp_asin = get_title_from_asin(asin_to_title, values['rating-drop-2'][0], get_title=False)
                    selected_reviews = sorted_recommended_reviews[tmp_asin]
                except Exception as e:
                    rev_exists = False
                    product_title = values['rating-drop-2'][0]
                    review_text_to_print = [f'No recommended reviews for: {product_title}']
                    print(f'No recommended reviews for: {product_title}')

                if rev_exists:
                    if len(selected_reviews) < num_to_print:
                        review_text_to_print = [r['reviewText'] for r in selected_reviews]
                    else:
                        review_text_to_print = [r['reviewText'] for r in selected_reviews[:num_to_print]]

                # print(f'sorted_recommended_reviews: {review_text_to_print} \n\n')
                    
                window['review-reader-0'].update(review_text_to_print)

                # Show Embedding Analysis
                exemplar_exists = True
                try:
                    exemplar_reviews = embed_sort_per_asin[tmp_asin]
                    # print(f'Exemplars {exemplar_reviews[:3]}')
                except Exception as e:
                    exemplar_text_to_print = [f'No exemplar reviews for: {product_title}']
                    print(f'No recommended reviews for: {exemplar_text_to_print}')
                
                if exemplar_exists:
                    if len(exemplar_reviews) < num_to_print:
                        exemplar_text_to_print = [r['reviewText'] for r in exemplar_reviews]
                    else:
                        exemplar_text_to_print = [r['reviewText'] for r in exemplar_reviews[:num_to_print]]
                
                window['review-reader-1'].update(exemplar_text_to_print)


            # elif event == 'Resample':
            #     updateChart()
            # elif event == '-Slider-':
            #     updateData(int(values['-Slider-']))
            #     # print(values)
            #     # print(int(values['-Slider-']))


        window.close()
    except Exception as e:
        sg.popup_error_with_traceback(f'While loop error.  Here is the info:', e)

if __name__=="__main__":
    main()