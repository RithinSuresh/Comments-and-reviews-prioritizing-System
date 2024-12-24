from flask import Flask, render_template, request, redirect, url_for
import json
from textblob import TextBlob
import pandas as pd
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import requests
from googleapiclient.discovery import build
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


path = "D:/Project/Project/static/"

app = Flask(__name__)
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        api_key = request.form['api_key']
        channel_id = request.form['channel_id']
        return redirect(url_for('processing', api_key=api_key, channel_id=channel_id))
    return render_template('index.html')

@app.route('/processing')
def processing():
    api_key = request.args.get('api_key')
    channel_id = request.args.get('channel_id')


    link = f'https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50'
    response = requests.get(link).text

    temp = json.loads(response)
    temp1 = temp["items"]
    Id = []

    for i in temp1:
        if i["id"]["kind"] == 'youtube#video':
            Id.append({"id":i["id"]["videoId"]})

    while True:
        try:
            if temp["nextPageToken"] is None or temp["nextPageToken"] == "":
                break
            else:
                nxtpg = temp["nextPageToken"]
                link = f'https://www.googleapis.com/youtube/v3/search?pageToken={nxtpg}&part=snippet&maxResults=25&order=relevance&q=site%3Ayoutube.com&topicId=%2Fm%2F02vx4&key=AIzaSyAGg1syggKzB6PtW8DvK6wyCol4UD0yTgw'
                response = requests.get(link).text

                temp = json.loads(response)
                temp1 = temp["items"]
                for i in temp1:
                    if i["id"]["kind"] == 'youtube#video':
                        Id.append({"id":i["id"]["videoId"]})
        except:
            break

    P_Id = Id
    code_lang = P_Id
    youtube = build('youtube', 'v3', developerKey=api_key)
    # box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]
    

    # for id_code in code_lang:
    #     def scrape_comments_with_replies():
    #         data = youtube.commentThreads().list(part='snippet', videoId=id_code['id'], maxResults='100', textFormat="plainText").execute()
    #         for i in data["items"]:
    #             name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
    #             comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
    #             published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
    #             likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
    #             replies = i["snippet"]['totalReplyCount']
                
    #             box.append([name, comment, published_at, likes, replies])
                
    #             totalReplyCount = i["snippet"]['totalReplyCount']
                
    #             if totalReplyCount > 0:
                    
    #                 parent = i["snippet"]['topLevelComment']["id"]
                    
    #                 data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent, textFormat="plainText").execute()
                    
    #                 for i in data2["items"]:
    #                     name = i["snippet"]["authorDisplayName"]
    #                     comment = i["snippet"]["textDisplay"]
    #                     published_at = i["snippet"]['publishedAt']
    #                     likes = i["snippet"]['likeCount']
    #                     replies = ""

    #                     box.append([name, comment, published_at, likes, replies])

    #         while ("nextPageToken" in data):
                
    #             data = youtube.commentThreads().list(part='snippet', videoId=id_code['id'], pageToken=data["nextPageToken"], maxResults='100', textFormat="plainText").execute()
                                                
    #             for i in data["items"]:
    #                 name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
    #                 comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
    #                 published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
    #                 likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
    #                 replies = i["snippet"]['totalReplyCount']

    #                 box.append([name, comment, published_at, likes, replies])

    #                 totalReplyCount = i["snippet"]['totalReplyCount']

    #                 if totalReplyCount > 0:
                        
    #                     parent = i["snippet"]['topLevelComment']["id"]

    #                     data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent, textFormat="plainText").execute()

    #                     for i in data2["items"]:
    #                         name = i["snippet"]["authorDisplayName"]
    #                         comment = i["snippet"]["textDisplay"]
    #                         published_at = i["snippet"]['publishedAt']
    #                         likes = i["snippet"]['likeCount']
    #                         replies = ''

    #                         box.append([name, comment, published_at, likes, replies])
                            
    #         # saves the scrapped comment in a datagrame then pushes it to a csv file
    #         df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box], 'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})
            
    #         sql_vids = pd.DataFrame([])

    #         # sql_vids = sql_vids.append(df, ignore_index = True)
    #         sql_vids = pd.concat([sql_vids, df], ignore_index=True)

    #         # sql_vids.to_csv('./youtube-comments.csv', index=False, header=False)
    #     try:
    #         scrape_comments_with_replies()
    #     except:
    #         continue

    # df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box], 'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})
        
    # sql_vids = pd.DataFrame([])

    # sql_vids = pd.concat([sql_vids, df], ignore_index=True)

    # data = sql_vids.copy()

    def scrape_comments_with_replies(youtube, video_id):
        box = []
        data = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults='100', textFormat="plainText").execute()

        while True:
            for item in data["items"]:
                comment = item["snippet"]['topLevelComment']["snippet"]
                box.append([
                    comment["authorDisplayName"],
                    comment["textDisplay"],
                    comment["publishedAt"],
                    comment["likeCount"],
                    item["snippet"]['totalReplyCount']
                ])

                if item["snippet"]['totalReplyCount'] > 0:
                    parent_id = item["snippet"]['topLevelComment']["id"]
                    replies_data = youtube.comments().list(part='snippet', maxResults='100', parentId=parent_id, textFormat="plainText").execute()
                    for reply in replies_data["items"]:
                        reply_snippet = reply["snippet"]
                        box.append([
                            reply_snippet["authorDisplayName"],
                            reply_snippet["textDisplay"],
                            reply_snippet['publishedAt'],
                            reply_snippet['likeCount'],
                            ''  # No replies to replies
                        ])

            if "nextPageToken" in data:
                data = youtube.commentThreads().list(part='snippet', videoId=video_id, pageToken=data["nextPageToken"], maxResults='100', textFormat="plainText").execute()
            else:
                break

        return pd.DataFrame(box, columns=['Name', 'Comment', 'Time', 'Likes', 'Reply Count'])
    
    all_comments = pd.DataFrame()
    for video in code_lang:
        try:
            df = scrape_comments_with_replies(youtube, video['id'])
            all_comments = pd.concat([all_comments, df], ignore_index=True)
        except:
            continue

    data = all_comments.copy()
    # data.to_csv('./youtube-comments.csv', index=False)


    def cleanTxt(text):
        text = str(text)
        text = re.sub(r'[^\w]', ' ', text)
        return text

    data['Comment'] = data['Comment'].apply(cleanTxt)

    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    # get polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    #Columns
    data['Subjectivity'] = data['Comment'].apply(getSubjectivity)
    data['Polarity'] = data['Comment'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0 :
            return 'Negative'
        elif score > 0:
            return 'Positive'
        else:
            return 'Neutral'
        
    data['Analysis'] = data['Polarity'].apply(getAnalysis)

    # data.to_csv('./youtube-comments.csv', index=False)

    data1 = data[["Comment"]]

    data1 = data1.drop_duplicates('Comment')
    data1['Comment'] = data1['Comment'].fillna('')

    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
    punc_set = set(punc)

    stop_words = list(ENGLISH_STOP_WORDS.union(punc_set))

    desc = data1['Comment'].values
    vectorizer = TfidfVectorizer(stop_words = stop_words)
    X = vectorizer.fit_transform(desc)
    words = vectorizer.get_feature_names_out()
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

    def tokenize(text):
        return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]
    
    kmeans = KMeans(n_clusters = 5, n_init = 20)
    kmeans.fit(X)
    Y = vectorizer.transform(data["Comment"].values)
    prediction = kmeans.predict(Y)

    c1,c2,c3,c4,c5=[],[],[],[],[]

    for (i,j) in zip(prediction,data["Comment"].values):
        if i == 0:
            c1.append(j)
        elif i == 1:
            c2.append(j)
        elif i == 2:
            c3.append(j)
        elif i == 3:
            c4.append(j)
        elif i == 4:
            c5.append(j)

    que = ["what", "why", "when", "where", "name", "how", "does", "which", "would", "could", "should", "has", "have", "whom", "whose", "question"]

    count = []
    count1 = 0
    for i in c1:
        z=i.lower()
        if any(q in z for q in que):
            count1+=1
    count.append(count1)

    count1=0
    for i in c2:
        z=i.lower()
        if any(q in z for q in que):
            count1+=1
    count.append(count1)

    count1=0
    for i in c3:
        z=i.lower()
        if any(q in z for q in que):
            count1+=1
    count.append(count1)

    count1=0
    for i in c4:
        z=i.lower()
        if any(q in z for q in que):
            count1+=1
    count.append(count1)

    count1=0
    for i in c5:
        z=i.lower()
        if any(q in z for q in que):
            count1+=1
    count.append(count1)


    count_list_pairs = list(zip(count, [c1, c2, c3, c4, c5]))
    # # Sort pairs by count in descending order
    sorted_pairs = sorted(count_list_pairs, key=lambda x: x[0], reverse=True)

    # # Unzip the pairs
    sorted_counts, sorted_lists = zip(*sorted_pairs)
    
    print("start")

    # pio.kaleido.scope.verbose = True
    # # # Sentiment Distribution Pie Chart
    # sentiment_distribution = pd.Series(data['Analysis']).value_counts()
    # print("1")
    # fig1 = px.pie(values=sentiment_distribution, names=sentiment_distribution.index, title='Sentiment Distribution')
    # print("1")
    # # image_bytes = to_image(fig1, format='png')
    # # with open('./fig1.png', 'wb') as f:
    # #     f.write(image_bytes)
    # pio.write_image(fig1,"./fig1.png")
    # print("1")
    # # Sentiment Frequency Bar Chart for Video
    # sentiment_frequency = pd.Series(data['Analysis']).value_counts().reset_index()
    # sentiment_frequency.columns = ['sentiment', 'frequency']
    # fig2 = px.bar(sentiment_frequency, x='sentiment', y='frequency', title='Frequency of Sentiment Types')
    # pio.write_image(fig2,"./fig2.png")
    # print("1")
    # # Likes vs. Sentiment Scatter Plot
    # fig3 = px.scatter(x=data['Likes'], y=data['Analysis'], color=data['Analysis'], title='Likes vs. Sentiment')
    # pio.write_image(fig3,"./fig3.png")
    # print("1")
    # # Convert Date column to datetime format
    # # data['Time'] = pd.to_datetime(data['Time'])

    # # # Upvotes and Downvotes Trend over Time Line Plot
    # # fig4 = px.line(data, x='Date', y=['Likes', 'Reply Count'], title='Likes and Reply Trend over Time')

    # # Upvotes vs. Downvotes Box Plot
    # fig5 = px.box(data, y=['Likes', 'Reply Count'], title='Upvotes vs. Downvotes Distribution')
    # pio.write_image(fig5,"./fig5.png")
    # print("1")

    data['Index'] = range(len(data))
    # analysis_trend = data.groupby(['Index', 'Analysis']).size().unstack().fillna(0)
    analysis_trend = data.groupby(['Index', 'Analysis']).size().unstack().fillna(0).iloc[:30]


    plt.figure(figsize=(10, 8))
    analysis_trend.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('Analysis Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.grid(True)
    plt.legend(title='Sentiment')
    plt.savefig(f'{path}/fig1.png')  # Save as PNG
    plt.close()

    plt.figure(figsize=(10, 8))
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
    plt.scatter(data['Index'], data['Likes'], c=data['Analysis'].map(colors), alpha=0.5)
    plt.title('Likes Distribution by Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Likes')
    plt.colorbar(label='Sentiment', ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: ['Negative', 'Neutral', 'Positive'][loc]))
    plt.grid(True)
    plt.savefig(f'{path}/fig2.png')  # Save as PNG
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(data['Polarity'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Polarity Distribution')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{path}/fig3.png')  # Save as PNG
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(data['Polarity'], data['Subjectivity'], alpha=0.5, color='purple')
    plt.title('Subjectivity vs. Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.grid(True)
    plt.savefig(f'{path}/fig4.png')  # Save as PNG
    plt.close()

    average_likes = data.groupby('Analysis')['Likes'].mean()
    plt.figure(figsize=(10, 8))
    average_likes.plot(kind='bar', color='orange')
    plt.title('Average Likes by Sentiment Category')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Average Likes')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.savefig(f'{path}/fig5.png')  # Save as PNG
    plt.close()

    # Simulate data - replace these with your actual data generation
    figs = ['fig1.png', 'fig2.png', 'fig3.png', 'fig4.png', 'fig5.png']
    categories = [
        f"Number of Questions is cluster 1 is: {sorted_counts[0]}"+json.dumps(sorted_lists[0][:31], indent = 2, ensure_ascii = False),
        f"Number of Questions is cluster 2 is: {sorted_counts[1]}"+json.dumps(sorted_lists[1][:16], indent = 2, ensure_ascii = False),
        f"Number of Questions is cluster 3 is: {sorted_counts[2]}"+json.dumps(sorted_lists[2][:16], indent = 2, ensure_ascii = False),
        f"Number of Questions is cluster 4 is: {sorted_counts[3]}"+json.dumps(sorted_lists[3][:16], indent = 2, ensure_ascii = False),
        f"Number of Questions is cluster 5 is: {sorted_counts[4]}"+json.dumps(sorted_lists[4][:16], indent = 2, ensure_ascii = False)
    ]


    # figs = ['fig1.png', 'fig2.png', 'fig3.png', 'fig4.png', 'fig5.png']
    # categories = [
    #     json.dumps(["string1", "string2"], indent=2),
    #     json.dumps(["string3", "string4"], indent=2),
    #     json.dumps(["string5", "string6"], indent=2),
    #     json.dumps(["string7", "string8"], indent=2),
    #     json.dumps(["string9", "string10"], indent=2)
    # ]
    return render_template('results.html', figures=figs, categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
