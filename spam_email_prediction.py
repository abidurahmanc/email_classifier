import pandas as  pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


df=pd.read_csv(r'mail_data.csv')

df.head()

df['Category'].value_counts()

df['Category']=df['Category'].replace('ham','not spam')

y=df['Category']
x=df['Message']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)

# Train the Multinomial Naïve Bayes model
mnb = MultinomialNB(alpha=0.5)
mnb.fit(X, y)

# Make predictions on a new text
new_text = ["buy today sell tomorrow", "congrats you won lottery"]
X_new = vectorizer.transform(new_text)
predictions = mnb.predict(X_new)

print("Predictions:", predictions)


import pickle
with open ('email_model.pkl','wb') as file:
    pickle.dump(mnb,file)

with open ('vectorizer.pkl','wb') as file:
    pickle.dump(vectorizer,file)

