import sys
import re

def clean_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # 1. Fix xi_i^_ -> \xi_i^*
    content = content.replace(r'\xi_i^_', r'\xi_i^*')

    # 2. Remove genui
    content = re.sub(r'genui.*?\n?', '', content)

    # 3. Fix image paths
    image_map = {
        'linear_regression_plot.png': 'lr.png',
        'knn_plot.png': 'knn.png',
        'lasso_plot.png': 'lasso.png',
        'elastic_net_plot.png': 'elastic.png',
        'ridge_plot.png': 'ridge.png',
        'pls_plot.png': 'pls.png',
        'catboost_plot.png': 'catboost.png',
        'ngboost_plot.png': 'ngboost.png',
        'huber_regressor_plot.png': 'huber.png',
        'gaussian_process_plot.png': 'gpr.png',
        'tree_plot.png': 'dt.png',
        'rf_plot.png': 'rf.png',
        'extra_trees_plot.png': 'et.png',
        'gbm_plot.png': 'gb.png',
        'xgboost_plot.png': 'xgb.png',
        'lgbm_plot.png': 'lgbm.png',
        'mlp_plot.png': 'mlp.png',
        'adaboost_plot.png': 'ada.png',
        'stacking_plot.png': 'st.png',
        'voting_plot.png': 'ensemble.png'
    }

    def replace_image(match):
        filename = match.group(2)
        mapped = image_map.get(filename, filename)
        return f'![{match.group(1)}](./{mapped})'
    
    content = re.sub(r'!\[(.*?)\]\(/home/tushar/.*?/scratch/(.*?)\)', replace_image, content)

    # 4. Fix multiline \boxed
    parts = content.split(r'\boxed{')
    out = parts[0]
    for part in parts[1:]:
        brace_count = 1
        idx = 0
        while brace_count > 0 and idx < len(part):
            if part[idx] == '{':
                brace_count += 1
            elif part[idx] == '}':
                brace_count -= 1
            idx += 1
        
        if brace_count == 0:
            inner = part[:idx-1]
            rest = part[idx:]
            
            # replace newlines with spaces inside inner
            inner = inner.replace('\n', ' ')
            # remove duplicate spaces
            inner = re.sub(r'\s+', ' ', inner)
            
            out += r'\boxed{' + inner.strip() + '}' + rest
        else:
            out += r'\boxed{' + part

    with open(path, 'w') as f:
        f.write(out)

if __name__ == '__main__':
    clean_file('presentation_slides.md')
    clean_file('README.md')
