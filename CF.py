'''

User-User (k-NN) Collaborative Filtering
-----------------------------------------
Author: Sundip Desai

Description:

Predict a user's rating based on a user-user notion vice
item-item notion.

Main function to be called is predictRating:

Inputs:
-------
item - a string that corresponds to the item the user would
       like a rating prediction

items - a list of items in string format

user - a dictionary containing the user in question ratings on all items.
       May contain null or None entries for items the user has
       not rated.

users - a list of user dictionaries that contain the respective ratings
        of items for each user


Outputs:
--------
prediction - predicted rating

'''

import math

def meanScore(a):

    '''
    meanScore
    ---------
    Description: Compute average score for a user

    Inputs: dictionary containing user's ratings

    Output: average rating (non-integer)
    '''
    s = sum([y for y in a.values() if y])
    n = len([x for x in a.values() if x])

    m = s/n

    return m


def pearsonCorrelation(items, user1, user2):

    '''

    pearsonCorrelation
    ------------------
    Description: Computes the similarity between two users via their ratings using
                 the Pearson correlation

    Inputs:
    -------
    items - list of items [string]
    user1 - dictionary of ratings for user in question
    user2 - dictionary of ratings for user that has rated the item that is in question

    Outputs:
    -------
    s - similarity value (can be integer or non-integer)

    '''
    n1 = 0
    n2 = 0
    n3 = 0

    for item in items:
        if user1[item] and user2[item]:
            n1 += (user2[item] - meanScore(user2)) * (user1[item] - meanScore(user1))
            n2 += (user2[item] - meanScore(user2)) ** 2
            n3 += (user1[item] - meanScore(user1)) ** 2

    s = n1 / (math.sqrt(n2) * math.sqrt(n3))

    return s

def cosineSimilarity(items, user1, user2):
    '''

    cosineSimilarity
    ------------------
    Description: Computes the similarity between two users via their ratings using
                 cosine similarity

    Inputs:
    -------
    items - list of items [string]
    user1 - dictionary of ratings for user in question
    user2 - dictionary of ratings for user that has rated the item that is in question

    Outputs:
    -------
    s - similarity value (can be integer or non-integer)

    '''
    c = 0
    c1 = 0
    c2 = 0
    for item in items:
        if user1[item] and user2[item]:
         c += user1[item]*user2[item]
         c1 += user1[item] ** 2
         c2 += user2[item] ** 2

    s = c / (math.sqrt(c1) * math.sqrt(c2))

    return s


def kNN(item, users):
    '''
    kNN
    ---
    Description: Nearest Neighbors

    Inputs:
    -------
    item - dictionary of item in question
    users - list of users (strings)

    Outputs:
    --------
    x - list of users that have rated the item in question

    Notes:
    -----
    Does not use k as an input. This will be added in next version

    '''


    x = []
    for user in users:
        if user[item]:
            x.append(user)

    return x


def predictRating(item, items, user, users):
    '''
    For description, inputs, outputs please go to top
    '''
    x = kNN(item, users)

    s = []
    if x:
        for x1 in x:
            s.append(pearsonCorrelation(items, user, x1))
    else:
        return 'There are no similar users! Please try another item-user combination.'

    num = []
    for i in range(len(s)):
        num.append(s[i]*(x[i][item] - meanScore(x[i])))

    prediction = meanScore(user) + (sum(num)/sum(map(abs, s)))

    return prediction

## -------- END OF CODE ---------- ##

'''
Movie Example
-------------

Customer Ratings for movies on a 5 star scale represented by Python dictionary.

Movies that haven't been rated is represented by 'None'.

'''

customer1 = {'Movie1': 4, 'Movie2': None, 'Movie3': 3, 'Movie4': 5}
customer2 = {'Movie1': None, 'Movie2': 5, 'Movie3': 4, 'Movie4': None}
customer3 = {'Movie1': 5, 'Movie2': 4, 'Movie3': 2, 'Movie4': None}
customer4 = {'Movie1': 2, 'Movie2': 4, 'Movie3': None, 'Movie4': 3}
customer5 = {'Movie1': 3, 'Movie2': 4, 'Movie3': 5, 'Movie4': None}

# Iterables
customers = [customer1, customer2, customer3, customer4, customer5]
movies = ['Movie1', 'Movie2', 'Movie3', 'Movie4']


# Sample Call
print(predictRating('Movie4', movies, customer3, customers))

















