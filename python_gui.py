from tkinter import *
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.tree import DecisionTreeClassifier

def getDomain(url):
	if re.search('www', url):
		start = url.find('www') + 3
		stop = url[start:].find('/') + start
		return url[start:stop]
	elif re.search('http://', url):
		start = url.find('http://') + 7
		stop = url[start:].find('/') + start
		return url[start:stop]
	elif re.search('https://', url):
		start = url.find('https://') + 8
		stop = url[start:].find('/') + start
		return url[start:stop]
	else:
		return 'no'

def having_ip_address(url):
    match = re.search(
'(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
'([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|' # IPv4
'((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url) # Ipv6
    if match:
        Label(text=f"haveing ip:{match} ").pack()
        return 1
    else:
        return 0
    
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        Label(text=f"abnormal :{match} ").pack()
        return 1
    else:  
        return 0
    
def count_www(url):
    url.count('www')
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                  r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                  r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                  r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|'
                  r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                  r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                  r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                  r'tr\.im|link\.zip\.net',url)
    if match:
        Label(text=f"shorning_sevice:{match} ").pack()
        return 1
    else:
        return 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    
    if match:
       Label(text=f"suspicous:{match} ").pack()
       return 1
    else:
        return 0
    
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
        return digits

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
            return letters
        
from tld import get_tld 

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
    
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
    
def has_php_path(url):
    if ".php" in url.lower():
        return 1
    else:
        return -1 
    
def find_com_domain(url):
    # Regular expression pattern to match URLs ending with .com
    pattern = r'.com'
    
    # Check if the pattern matches the URL
    if re.search(pattern, url):
        return 1
    else:
        return -1
def find_org_domain(url):
    # Regular expression pattern to match URLs ending with .com
    pattern = r'.org'
    
    # Check if the pattern matches the URL
    if re.search(pattern, url):
        return 1
    else:
        return -1

def checkSubdomains(domain):
	'''if len(domain.split('.')) > 3:
		return -1
	elif len(domain.split('.')) == 3:
		return 0
	else:
		return 1'''
	return len(domain.split('.'))

def main(url):
    domain = getDomain(url)
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
   # status.append(prefixSuffix(domain))
    status.append(has_php_path(url))
   # status.append(checkTags(url, soup, domain))
    #status.append(checkAnchors(url, soup, domain))
    status.append(find_com_domain(url))
    status.append(find_org_domain(url))
   # status.append(Download_URL(url))
   # status.append(Free_URL(url))
    #status.append(url_zip_rar(url))
    #status.append(checkServerForm(url, soup, domain))
   # status.append(checkSSL(url))
    status.append(checkSubdomains(domain))
   # Label(text=f"accurancy:{domain} ").pack()
    print(status)
    
    return status

def show_confusion_matrix():
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred_rf), index=['benign', 'defacement', 'phishing', 'malware'],
                         columns=['benign', 'defacement', 'phishing', 'malware'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt=".1f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

def get_prediction_from_url(test_url):
    features_test = main(test_url)
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))
    pred = tree1.predict(features_test)
    if int(pred[0]) == 0:
        res="THIS IS PHISHING WEBSITE"
        return res
    elif int(pred[0]) == 1.0:
        res="THIS IS SAFE WEBSITE"
        return res
    elif int(pred[0]) == 2.0:
        res="PHISHING"
        return res
    elif int(pred[0]) == 3.0:
        res="SAFE"
        return res
def getvals():
    url = uservalue.get()
    prediction = (get_prediction_from_url(url))
    print(prediction)
    Label(text=f"This URL is :{prediction} ").pack()
    s = Label(text="test acurancy accurancy: %0.3f" % score1).pack()
    prediction_on_traning_data = tree1.predict(X_train)
    accuracy_on_traning_data = accuracy_score(y_train, prediction_on_traning_data)
    print("accuracy: %0.3f" % accuracy_on_traning_data)
    Label(text=f"accuracy :{accuracy_on_traning_data} ").pack()
'''class App(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.entryhingy = Entry
        self.entryhingy.pack()
        self.contents = StringVar()
        self.contents.set("This is a variable")
        self.entryhingy["textvarible"] = self.contents
        self.entryhingy.bind('<Key-Return>', self.print_contents)
    def print_contents(self, event):
        print("Hi. The current entry content is :",self.contents.get())'''
def harry(event):
    print(f"udfndfsd {event.x}")
a = Tk()

a.geometry("844x644")

a.minsize(200,200)
a.maxsize(844,644)
a.title("Phishinig Detection")



df = pd.read_csv('C:/Users/DELL/Desktop/coding/project/Project.csv')
X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
    'count_dir', 'count_embed_domian', 'short_url', 'count-https',
    'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
    'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
    'count-letters','Has PHP Pages','COM Domain','ORG Domain','Subdomains']]
print(X.shape)
X.head()
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df["type-code"] = lb_make.fit_transform(df["type"])
y = df['type-code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#rf = RandomForestClassifier(n_estimators=10,max_features='sqrt',criterion="entropy")
#rf.fit(X_train, y_train)
#y_pred_rf = rf.predict(X_test)
#print(classification_report(y_test, y_pred_rf, target_names=['benign', 'defacement','phishing','malware']))
#score = accuracy_score(y_test, y_pred_rf)
#print("accuracy: %0.3f" % score)
tree1 = DecisionTreeClassifier(max_depth=3, random_state=0)
tree1.fit(X_train, y_train)
y_pred_rf = tree1.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=['benign', 'defacement','phishing','malware']))
score = tree1.score(X_train, y_train)
score1 = tree1.score(X_test, y_test)
print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, y_test)))
#widget = Button(a, text="click me please")
#widget.pack()
#widget.bind('<Button-1>', harry)
t= Label(text=classification_report(y_test, y_pred_rf, target_names=['benign', 'defacement','phishing','malware'])).pack()
s= Label(text="training accurancy: %0.3f" %score ).pack()
s= Label(text="test acurancy accurancy: %0.3f" %score1 ).pack()
#button = ttk.Button(a, text="Show Confusion Matrix", command=show_confusion_matrix)
#button.pack(pady=20)
#Label(a, text="hello", font="comicsansns 13 bold").grid(row=0, column=3)

uservalue = StringVar()

userentry = Entry(a,textvariable=uservalue)

userentry.pack()
Button(text="Submit",command=getvals).pack()

#gui logic'''
a.mainloop()

'''def getvals():
    print(uservalue.get())

#photo = PhotoImage(file="C:\\Users\\DELL\\Desktop\\coding\\project\\Screenshot 2023-11-04 212946.png")
#image = Image.open("C:\\Users\\DELL\\Pictures\\New folder\\beautiful-photography-ganpati-statue-with-1830309.jpg")
#photo = ImageTk.PhotoImage(image)
#pic = Label(image=photo)
#pic.pack()

#s = Label(text="hello world",bg="red",fg="white",padx=24,pady=56,font="comicsansms 19 bold",borderwidth=3,relief=SUNKEN)
#s = Frame(a,bg="grey",borderwidth=6,relief=SUNKEN)
#s.pack(side=LEFT,anchor="nw")
#b1 = Button(s,fg="red", text="print now",command=t)
#b1.pack()

url = Label(a, text="Enter the URL")

url.grid()
frm = ttk.Frame(a, padding=10)
frm.grid()
ttk.Label(frm, text="hello world").grid(column=0, row=0)
ttk.Button(frm, text="Quit",command=a.destroy).grid(column=1, row=0)

btn = ttk.Button(frm, text="hello")
print(btn.config().keys())
print(dir(frm))
print(set(dir(frm)))
#classes in tkinter 
#booleanvar,Doublevar,Intvar,Stringvar

uservalue = StringVar()

userentry = Entry(a,textvariable=uservalue)

userentry.grid(row=0,column=1)
Button(text="Submit",command=getvals).grid()'''


#Text = adda the text
#bd = Background
#fg= foregroung
#font = sets the font
#padx = x padding
#pady = y padding
#relief = border styling = SUKEN, RAISED , GROOVE, RIDGE