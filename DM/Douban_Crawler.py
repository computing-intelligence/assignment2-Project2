#encoding=utf8
import requests
import bs4 as bs
import pandas as pd

#通过URL获取相应的soup
def get_soup(url):
	res=requests.get(url)
	soup=bs.BeautifulSoup(res.text,"html.parser")
	return soup

#获取正在热映的电影主页网址列表，以及每部电影关联的推荐电影
def list_movies_url():
	list_url=[]
	url_playing_list='https://movie.douban.com/cinema/nowplaying'
	a_items=get_soup(url_playing_list).find('ul',{'class':'lists'}).find_all('li',{'class':'poster'})
	def url_cut(url):return url[:url.find('?')]#去掉问号后面的无用字符串，例如https://movie.douban.com/subject/26430636/?from=playing_poster只保留问号之前的串
	for k in a_items:list_url.append(url_cut(k.find('a').get('href')))

	for root_url in list_url:#根据根节点电影URL找对应推荐的子节点电影URL
		try:
			recommend_movies = get_soup(root_url).find('div', {'class': 'recommendations-bd'}).find_all('dl', {'class': ''})
			for k in recommend_movies:
				url = url_cut(k.find('a').attrs['href'])#这种写法与前面的find('a').get('href')等效
				list_url.append(url)
		except:#有的电影没有对应的推荐电影，findall会报错，则直接跳过
			continue
		list_url=list(set(list_url))#去除重复值
	return list_url

#抓取电影评论（输入电影主页URL和所需评论条数，最多200条。前提是输入的主页网址是干净的，不带后缀）
def list_comment(url_movie,num):
	url_comments=url_movie+'comments?start=0&limit=20&sort=new_score&status=P&percent_type='#从电影主页URL转换到更多评论页URL
	start=0#int(url_comments[url_comments.find('=')+1:url_comments.find('&'):])#抓取当前start数值
	comment_text=[]
	comment_votes=[]
	comment_rating=[]
	while(num>0):
		comment_div_all=get_soup(url_comments).find_all('div',{'class':'comment'})
		for i in range(len(comment_div_all)):
			# 以下为爬取第i条评论的文本内容
			comment_text.append(comment_div_all[i].find('span',{'class':'short'}).getText())
			# 以下为爬取第i条评论对应的其他人觉得这条评论“有用”的数字，如出现空值等异常则getText()方法会报错，所以直接append零值
			try:
				comment_votes.append(comment_div_all[i].find('span', {'class': 'votes'}).getText())
			except:
				comment_votes.append('0')
			#以下为爬取第i条评论对应的用户给出的星级打分，共有10、20-50共五种，代表“很差”、“较差”至“力荐”，原文样例：span class="allstar50 rating为“力荐”
			k = str(comment_div_all[i].find_all('span', {'class': 'comment-info'}))
			if k.find('allstar'): comment_rating.append(k[(k.find('allstar') + 7):(k.find('allstar') + 8)])

		num-=20
		start+=20
		url_comments=url_comment_page(url_comments,start)
	return comment_text[:num],comment_votes[:num],comment_rating[:num]

#根据start的值生成相应分页评论的URL
def url_comment_page(url_comments,start_num):
	url_front=url_comments[:url_comments.find('=')+1]#页面start编号之前的字符串
	url_behind=url_comments[url_comments.find('&'):]#页面start编号之后的字符串
	return url_front+str(start_num)+url_behind

#热门电影URL列表抓取
comment_num_per_film=199#每部电影抓取多少条评论
list_url=list_movies_url()
from collections import Counter
url_list_counts=sorted(Counter(list_url), reverse=True)
df_all=pd.DataFrame()#建立一个空的DataFrame
for url in url_list_counts:#每一轮循环是一部电影
	comment_text_votes_rating = list_comment(url, comment_num_per_film)  # 这个tuple由3个list组成，0、1、2分别代表text、votes和rating
	comment_text = comment_text_votes_rating[0]
	comment_votes = [int(k) if k != ' ' and k !='' else 0 for k in comment_text_votes_rating[1]]  #原来是String，转成int型，空值和空格值替换为0
	comment_rating = [int(k) if k != ' ' and k !='' else 0 for k in comment_text_votes_rating[2]] #原来是String，转成int型，空值和空格值替换为0
	#将这部电影的comment_num_per_film个评论放入DataFrame中
	df_data_per_film={'text':comment_text,'rating':comment_rating,'url':[url for k in range(len(comment_text))]}#建立字典类型数据，为转成DataFrame做准备
	df_per_film=pd.DataFrame(df_data_per_film)
	df_all=df_all.append(df_per_film,ignore_index=True)

# 评论文本中有\r符号会使得Excel输出中出现换行，所以用空值替换删除此符号
for k in range(len(df_all['text'])):
	if df_all.loc[k, 'text'].find('\r') != -1: df_all.loc[k, 'text'] = df_all.loc[k, 'text'].replace('\r', '')

#将爬取的内容存入本地文件中
df_all.to_csv('C:/Users/trans02/PycharmProjects/untitled1/pycode/Sentiment Analysis/text_rating.csv',encoding='gb18030')