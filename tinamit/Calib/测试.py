#!/usr/bin/env python
# -*- coding: utf-8 -*-

# people = ["Hanna","Louisa","Claudia", "Angela","Geoffrey", "aparna"]
# # DON'T TOUCH THIS PLEASE!
#
# #Change "Hanna" to "Hannah"
# people[0] = 'Hannah'
#
# #Change "Geoffrey" to "Jeffrey"
# people[-2] = 'Jeffrey'
# #Change "aparna" to "Aparna" (capitalize it)
# people[-1] = people[-1].upper()
# print("this is gaby's: ", people[-1])

# name =  ['Elie', 'Tim', 'Matt']
# answer = [fl[0] for fl in name]
# print(answer)
#
# # nums = [1,2,3,4,5,6]
# # answer2 = []
# # [nums.join(num) if num%2 == 0 else nums.remove(num) for num in nums]
# # print(nums)
#
# answer2 = [num for num in [1,2,3,4,5,6] if num%2 == 0]
# print(answer2)
#
# l1 = [1,2,3,4]
# l2 = [3,4,5,6]
# l1.extend(l2[2:])
# print(l1)
# num = []
# num.extend(l1[2:4])
# print(num)
# answer2 = [name[::-1].lower() for name in ['Elie', 'Tim', 'Matt']]
# print(answer2)

# answer = [num for num in l1 and l2 if num == 3 and 4 in l1 and l2]
# print(answer)

# list = [[1,2,3], [3,4,5], [5,6,7]]
# for l in list:
# 	for num in l:
# 		print(num)
#
# gabe = [[t for t in range(1,4)] for a in range(1,5)]
# pd = [['Gaby' if a%2 == 0 else '蛋' for a in range(1,6)] for g in range(1,2)]
# print(gabe, '\n', pd)

# This code picks a random food item:
# from random import choice  # DON'T CHANGE!
# food = choice(["cheese pizza", "quiche", "morning bun", "gummy bear", "tea cake"])  # DON'T CHANGE!
# # DON'T CHANGE THIS DICTIONARY EITHER!
# bakery_stock = {
# 	"almond croissant": 12,
# 	"toffee cookie": 3,
# 	"morning bun": 1,
# 	"chocolate chunk cookie": 9,
# 	"tea cake": 25}
#
# if food in bakery_stock:
#     print(f'{bakery_stock[food]} left')
# else:
#     print("we don't make it")
#
#
# # quantity = bakery_stock.get(food)
# # if quantity:
# #     print(f"{quantity} left")
# # else:
# #     print("we don't make that")

# playlist = {'title':'patagonia bus',
#             'author': 'Gaby Peng',
#             'song': [{'title':'s1', 'artist':['pidan'], 'duration': 2.5},
#                      {'title':'s2', 'artist':['Jay'], 'duration': 3},
#                      {'title':'s3', 'artist':['Adamowski'], 'duration': 2}
#                      ]
#             }
# total_len = 0
# for song in playlist['song']:
#     total_len += song['duration']
# print(total_len)


# num = dict(fst=1, snd=3, trd=9)
# s_num = {k: v ** 2 for k, v in num.items()}
# print(s_num)
#
# num2 = {num: num /5 for num in range(1, 101, 4)}
# print(num2)
#
# jay1 = ['JAY', 'JAY', 'JAY']
# jay2 = ['傻', '死', '了']
# # truth = {jay1[i]: jay2[i] for i in range(0, 3)}
#
# truth = dict(zip(jay1, jay2))
# print(truth)
# # num = [1,2,3,4]
# # new_num = {num: ('odd' if num % 2 != 0 else 'even') for num in num}
# # print(new_num)
#
# person = [["name", "Jared"], ["job", "Musician"], ["city", "Bern"]]
# list = dict(person)
# print(list)

# def sing_happy_bday():
#     print('happy bday')

# def add(a,b):
#     return a+b
# def sub(a,b):
#     return a-b
# def math(a,b, fn=add):
#     return fn(a,b)
# print(math(2,2))
# print(math(2,2, sub))


# def single_letter_count(a, b):
#     if b in a:
#         return a.count(b)
#     return 0
# print(single_letter_count('hello', 'z'))
#
# def single_letter_count(string,letter):
# #     return string.lower().count(letter.lower())
# def list_manipulation(l, cmd, loc, val=None):
#     if cmd == "remove" and loc == "end":
#         l.pop()
#         return l
#     elif cmd == "remove" and loc == "beginning":
#         l.pop(0)
#         return l
#     elif cmd == "add" and loc == "beginning":
#         l.insert(0, val)
#         return l
#     elif cmd == "add" and loc == "end":
#         l.append(val)
#         return l
# print(list_manipulation([1,2,3], "add", "beginning", 20))
# print(list_manipulation([1,2,3], "remove", "beginning"))

# def multiply_even_numbers(a):
#     mul = 1
#     for num in a:
#         if num % 2 == 0:
#             mul = mul * num
#     return mul
# print(multiply_even_numbers([2,3,4,5,6]))

# def compact(a):
#     b = []
#     for i in a:
#         if i:
#             b.append(i)
# #     return b
# def compact(l):
#     return [val for val in l if val]
# # print(compact([0,1,2,"",[], False, {}, None, "All done"])


# operation = ['add', 'substract', 'multiply', 'divide']
# def calculate(**kwargs):
#
#     operation2= {'add': kwargs.get('first') +  kwargs.get('second'),
#                  'substract': kwargs.get('first') - kwargs.get('second'),
#                  'multiply': kwargs.get('first') * kwargs.get('second'),
#                  'devide':kwargs.get('first') /  kwargs.get('second')
#                  }
#     is_float = kwargs.get('make_float', False)
#     operation1 = operation2[kwargs.get('operation')]
#     if is_float:
#         result = f"{kwargs.get('message', 'the result is')} {float(operation1)}"
#     else:
#         result = f"{kwargs.get('message', 'the result is')} {int(operation1)}"
#     return result
# print(calculate(make_float=True, operation='devide', first=2, second=4))
#
# print(sys.getsizeof([i*20 for i in range(5000)]))
# print(sys.getsizeof(i*20 for i in range(5000)))

# for i in ['a', 'b', 1]:
#     print(type(i))
#
# def interleave(a, b):
#     c = list(zip(a, b))
#     print(c)
#     d = [''.join(i) for i in c]
#     print(d)
#     e = ''.join(d)
#
#     return print(e)
# interleave('gaby', 'pidan')
# lists = {"small":2, "big":22}
# def sum(small, big):
#     count = small + big
#     print(count)
# sum(**lists)
#
# param_name =  ["Ptq - s1", "Ptr - s1", "Kaq - s1", "Peq - s1", "Pex - s1", "Ptq - s2", "Ptr - s2",
#                         "Kaq - s2", "Peq - s2", "Pex - s2", "Ptq - s3", "Ptr - s3", "Kaq - s3", "Peq - s3", "Pex - s3",
#                         "Ptq - s4", "Ptr - s4", "Kaq - s4", "Peq - s4", "Pex - s4",
#                         "POH Kharif", "POH rabi", "Capacity per tubewell"
#                 ]

# THIS IS FOR FAST SA RESULT ANALYSIS
# result = []
# with open('D:\\Thesis\\pythonProject\\Tinamit\\tinamit\\Calib\\SA_algorithms\\Fast\\Simulation\\Fast_sensitivity_ave_n_last.out') \
#     as f:
#     for line in f:
#         result.append(json.loads(line))
# f.close()


# for i in range(len(result[0][0])): #result[0][0] is a; len(result[0][0]) = 11
#     print(f'this is {i+1}th class for A')
#     paramlst = []
#     for p in result[0][0][i]['S1', 'ST']:
#         paramlst.append(p)
#     # for t in result[0][0][i]['ST']:
#     #     paramlst.append(t)
#     # Si < 0.01 & St < 0.1
#     tmp_non_influential = [ for p in paramlst if paramlst[0] < 0.01 and paramlst[1] < 0.1]
#         tmp_non_influential
#
#     paramdict = dict(zip(param_name, paramlst))
#     # tmp = map(lambda k, v: paramdict.update({k: v}), param_name, paramlst)
#     # print(paramdict)
#
#     sort_param_order = sorted(paramdict.items(), key=lambda x: x[1], reverse=True)
#     print(sort_param_order)

# noninfluential_param_of_ave_last = []
# influential_param_of_ave_last = []
# for a_or_b in result[0]: # a_or_b is o (s1*23+st*23) or 1 (s1*23+st*23)
#     non_ave_last = []
#     ave_last = []
#     for i in a_or_b: #i = 11 classes
#         non_a_l= [] #[{param_name1: (s1, st)}, {param_name2: (s1, st)}] a list with 23 dicts
#         a_l = []
#         for j in range(23): #j = 23
#             if i['S1'][j] < 0.01 and i['ST'][j] < 0.1:
#                 #print(fa.problem['names'][j], (i['S1'][j], i['ST'][j] ))
#                 non_a_l.append({param_name[j]: (i['S1'][j], i['ST'][j])})
#             else:
#                 a_l.append({param_name[j]: (i['S1'][j], i['ST'][j])})
#         non_ave_last.append(non_a_l)
#         ave_last.append(a_l)
#     noninfluential_param_of_ave_last.append(non_ave_last)
#     influential_param_of_ave_last.append(ave_last)

# for i, a_or_b2 in enumerate(noninfluential_param_of_ave_last): # i=0 a, i=1 b; i=0 -> 0 -> len()
#     for j, cat in enumerate(a_or_b2):

#         # print(f'{i+1}', f'{j+1} \n non:{len(cat)}', cat)
# all_sorted_class_4_ab = []
# for i, aob in enumerate(influential_param_of_ave_last): # i = 0 or 1; a_or_b : len()=11
#     sorted_class = []
#     for j, cat in enumerate(aob): # j = 11, cat = [{}, {}, ...]
#         key = []
#         val = []
#
#         # print(cat)
#         # sorted_parameters = sorted(cat, key = lambda x: x.items())
#         # print(sorted_parameters)
#
#         # for l in range(len(cat)):0
#         sorted_parameters = []
#         for k, v in [(k, v) for x in cat for (k, v) in x.items()]: #
#             key.append(k)
#             val.append(v)
#         cat2 = dict(zip(key, val))
#         cat22 = sorted(cat2.items(), key=lambda x: x[1][1], reverse=True) #sorted cat2
#         sorted_class.append(cat22)
#     all_sorted_class_4_ab.append(sorted_class)
# print(sorted_class)


# print(all_sorted_class_4_ab)
# for i, cont in enumerate(all_sorted_class_4_ab):
#     print(f'parameter {i}')
#     for j, classes in enumerate(cont):
#         print(f'class {j}{classes}')
#         # for items in classes:
#     print(f'{items}')

# print(f'{i+1}', f'{j+1} \n influential:{len(cat)}', cat22)

#     sorted(cat2, key=lambda k: k['name'])
# cat2 = {}
# for d in cat:
#     cat2.extend(**d)
#         # (sorted(d.items(), key=lambda x: x[1][1], reverse=True))
# print(f'{i+1}', f'{j+1} \n influential:{len(cat)}', d)


# THIS IS FOR MORRIS RESULT ANALYSIS
# result = []
# with open('D:\\Thesis\\pythonProject\\Tinamit\\tinamit\\Calib\\SA_algorithms\\Morris\\morris_sensitivity_3.out') \
#     as f:
#     for line in f:
#         result.append(json.loads(line))
# f.close()
#
# for i in range(len(result[0][1])): #result[0][0] is a; len(result[0][0]) = 11
#     print(f'this is {i+1}th class for B')
#     paramlst = []
#     for p in result[0][1][i]['mu_star']:
#         paramlst.append(p)
#     paramdict = dict(zip(param_name, paramlst))
#     # tmp = map(lambda k, v: paramdict.update({k: v}), param_name, paramlst)
#     # print(paramdict)
#
#     sort_param_order = sorted(paramdict.items(), key=lambda x: x[1], reverse=True)
#     print(sort_param_order)


# trend_ab.append(result[0]['Trend parameters'])
# pairs = dict(zip(param_name, ))

# r2 = [1.344643,2.342352,3.3253,4,5]
# residuals = [2,3,4,5,7.643543]
# r22 = [round(i,2) if isinstance(i, float) else i for i in r2]
# residuals2 = [round(i,2) if isinstance(i, float) else i for i in residuals]
# # r22 = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, r2))
# # r22 = [[round(i,2) if isinstance(i, float) else i for i in x ] for x in r2]
# # residuals2 = tuple(map(lambda x: isinstance(x[i], float) and round(x, 2) or x, residuals))
# print({'R2': r22, 'residuals': residuals2})

# {\n"3": 1, \n"4": 2\n}

# result = [] #[[{'r2':[], 'residuals':[], }, {}, ...{}]]
# with open('D:\\Thesis\\pythonProject\\Tinamit\\tinamit\\Calib\\SA_algorithms\\Morris\\residual_result_4_polys_winter.out') \
#     as f:
#     for line in f:
#         result.append(json.loads(line))
# f.close()
#
# good_r2 = []
#
# for i in range(215):
#     count = len([v for v in result[0][i]['R2'] if v > 0.6]) # all season is 'r2'
#     print(f'count{i}: ', round(count / 1200, 2))
#     good_r2.append({'goodr2': count})
# print(good_r2)

# for j in range(1200):
#     totalcount = []
#     for i in range(215):
#         count = len([v for v in result[0][i]['r2'] if v > 0.6])
#         print(f'count{i}: ', round(count/1200, 2))
#     totalcount.append({'goodr2': [count]})
# good_r2.append(totalcount)
# print(good_r2)

'''
HTTP REQUEST EXCERSISE
'''

# import requests
# from termcolor import colored
# from pyfiglet import figlet_format
# from random import choice
# welcoming1 = figlet_format("DAD'S OLD JOKE")
# wel0 = 'Lets Begin'
# welcoming2 = colored(welcoming1, color='cyan')
# welcoming22 = colored(wel0, color='yellow', attrs=['blink', 'dark', 'bold', 'underline'])
# print(welcoming2)
# print(welcoming22)
#
# si = input('what would u like to search for? ')
# url = "https://icanhazdadjoke.com/search"
# # res = requests.get(url, headers={'Accept': "text/plain"})
# res = requests.get(url, headers={'Accept': "application/json"}, params={'term': si}).json()
#
# num_jokes = res['total_jokes']
# results = res['results']
# if num_jokes > 1:
# 	print(f"there are many jokes, we found {res['total_jokes']} jokes for you, here's the one")
# 	ran = choice(results)
# 	print(ran['joke'])
# elif num_jokes == 1:
# 	print(f'theres only one Joke is {(results[0])}')
# else:
# 	print('there are no jokes')

# class Chicken:
# 	total_eggs = 0
#
# 	def __init__(self, species, name):
# 		self.species = species
# 		self.name = name
# 		self.eggs = 0
#
# 	def lay_egg(self):
# 		self.eggs += 1
# 		Chicken.total_eggs += 1
# 		return self.eggs
# 	def __repr__(self):
# 		return "I'm a chicken JAY"
# c1 = Chicken('s1', 'a')
# c2 = Chicken('s2', 'b')
# print(c1)
# print(Chicken.total_eggs) #0
# print(c1.lay_egg()) #1
# print(Chicken.total_eggs) #1
# print(c2.lay_egg()) #2
# # c2.lay_egg()
# print(c2.lay_egg())
# print(Chicken.total_eggs)

# class Card:
# 	def __init__(self, suit, value):
# 		self.suit = suit
# 		self.value = value
# 	def __repr__(self):
# 		return f"{self.value} of {self.suit}"
#
# class Deck:
# 	'''
# 	should have a cards attribute with all 52 possible instances of Card .
# 	'''
# 	def __init__(self):
# 		self.cards = []
#
#
# 	def count(self, remaining_card): #returns a count of how many cards remain in the deck.
#
# 	def _deal(self):
# 		'''
# 		accepts a number and removes at most that many cards from the deck
# 		(it may need to remove fewer if you request more cards than are currently in the deck!).
# 		If there are no cards left, this method should return a ValueError: "All cards have been dealt".
# 		:return:
# 		'''
#
# 	def shuffle(self):
# 		'''
# 		shuffle a full deck of cards.
# 		If there are cards missing from the deck, return a ValueError:"Only full decks can be shuffled".
# 		:return: shuffled deck.
# 		'''
#
# 	def deal_card(self):
# 		'''
# 		uses the _deal  method to deal a single card from the deck and return that single card.
# 		:return:
# 		'''
# 	def deal_hand(self):
# 		'''
# 		accepts a number and uses the _deal method to deal a list of cards from the deck
# 		:return:  return that list of cards.
# 		'''
# 	def __repr__(self):
# 		return f"Deck of {self.} cards"
