import re
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import lightgbm as lgb

df=pd.read_csv('C:/Users/DELL/Desktop/coding/project/Project.csv')
print(df.shape)
df.head()
def having_ip_address(url):
    match = re.search(
'(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
'([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|' # IPv4
'((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url) # Ipv6
    if match:
        return 1
    else:
        return 0
df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))
from urllib.parse import urlparse
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:  
        return 0
df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df['google_index'] = df['url'].apply(lambda i: google_index(i))
def count_dot(url):
    count_dot = url.count('.')
    return count_dot
df['count.'] = df['url'].apply(lambda i: count_dot(i))
def count_www(url):
    url.count('www')
    return url.count('www')
df['count-www'] = df['url'].apply(lambda i: count_www(i))
def count_atrate(url):
    return url.count('@')
df['count@'] = df['url'].apply(lambda i: count_atrate(i))
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')
df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
'tr\.im|link\.zip\.net',
url)
    if match:
        return 1
    else:
        return 0
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))
def count_https(url):
    return url.count('https')
df['count-https'] = df['url'].apply(lambda i : count_https(i))
def count_http(url):
    return url.count('http')
df['count-http'] = df['url'].apply(lambda i : count_http(i))
def count_per(url):
    return url.count('%')
df['count%'] = df['url'].apply(lambda i : count_per(i))
def count_ques(url):
    return url.count('?')
df['count?'] = df['url'].apply(lambda i: count_ques(i))
def count_hyphen(url):
    return url.count('-')
df['count-'] = df['url'].apply(lambda i: count_hyphen(i))
def count_equal(url):
    return url.count('=')
df['count='] = df['url'].apply(lambda i: count_equal(i))
def url_length(url):
    return len(str(url))

df['url_length'] = df['url'].apply(lambda i: url_length(i))

def hostname_length(url):
    return len(urlparse(url).netloc)
df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))
df.head()
def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
        return digits
df['count-digits']= df['url'].apply(lambda i: digit_count(i))
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
            return letters
df['count-letters']= df['url'].apply(lambda i: letter_count(i))

from urllib.parse import urlparse
from tld import get_tld 

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))

df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))

from sklearn.preprocessing import LabelEncoder,StandardScaler
lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])

#X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
##'count_dir', 'count_embed_domian', 'short_url', 'count-https',
#'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
#'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
#'count-letters']]
#print(X)
#y = df['type_code']
#print(y)

#params = {
#    'objective': 'multiclass',
    #'num_class': 4,  
    #'metric': 'multi_logloss',
   # 'num_leaves': 31,
  #  'learning_rate': 0.05,
 #   'feature_fraction': 0.9
#    }
#train_data = lgb.Dataset(X_train, label=y_train)
#test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
#num_round = 100
#bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
#y_pred_lgbm_prob = bst.predict(X_test)
#score_lgbm = accuracy_score(y_test, y_pred_lgbm)
#print("LightGBM accuracy:", score_lgbm)

def main(url):
    status = []
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)
    status.append(tld_length(tld))
    print(status) 
    return status
 
def get_prediction_from_url(test_url):
    
  #  from sklearn.preprocessing import LabelEncoder
  #  lb_make = LabelEncoder()
  #  df["type_code"] = lb_make.fit_transform(df["type"])

    X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
    'count_dir', 'count_embed_domian', 'short_url', 'count-https',
    'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
    'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
    'count-letters']]
    print(X)
    y = df['type_code']
    print(y)
    features_test = main(test_url)  # Assuming main function extracts features
    features_test = np.array(features_test).reshape((1, -1))
    print(features_test)
    X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(X_train, X_test, y_train, y_test)

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=['benign', 'defacement', 'phishing', 'malware']))
    score = accuracy_score(y_test, y_pred_rf)
    print("Random Forest accuracy: %0.3f" % score)
  #  bst = lgb.LGBMClassifier(random_state=42)  # You need to specify your model parameters here
   # bst.fit(X_train, y_train)
  #  lgbm = lgb.LGBMClassifier(random_state=42)  # Add model parameters as needed
  #  lgbm.fit(X_train, y_train)
  #  print(classification_report(y_test, y_pred_lgbm, target_names=['benign', 'defacement', 'phishing', 'malware']))
  #  print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))
  #  pred = rf.predict(features_test)
   # bst_pred=bst.fit(X_train, y_train)
   # pred = bst_pred.predict(features_test)
   # print("Predicted category:", pred)
    params = {
    'objective': 'multiclass',
    'num_class': 4,  
    'metric': 'multi_logloss',
   'num_leaves': 31,
   'learning_rate': 0.05,
   'feature_fraction': 0.9
     }
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
    y_pred_lgbm_prob = bst.predict(X_test)
    y_pred_lgbm = [int(x.argmax()) for x in y_pred_lgbm_prob]
    score_lgbm = accuracy_score(y_test, y_pred_lgbm)
    print("LightGBM accuracy:", score_lgbm)
    pred = bst.predict(features_test)
    print(pred)
    pred = int(pred[0])
    print(pred)
    if pred == 0:  
        return "SAFE"
    elif pred == 1:
        return "DEFACEMENT"
    elif pred == 2:
        return "PHISHING"
    elif pred == 3:
        return "MALWARE"

urls = ['titaniumcorporate.co.za', ]
for url in urls:
    print(get_prediction_from_url(url))


