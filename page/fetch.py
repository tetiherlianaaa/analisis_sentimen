import streamlit as st
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import urlopen

with open ('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
	return fetched_text

def selection(key):
	option = st.selectbox('How would you like to provide the data?',('URL', 'Paste/Write Text'), index=1, key=key)
	st.write('You selected:', option)
	if option == 'Paste/Write Text' :
		message = st.text_area("Enter Text", "Type Here ..", key=key+'text')
		return (0,message)
	else:
		url = st.text_area("Enter URL", "Paste Here ..", key= key+'url')
		return (1,url)

def front_up():
    html_temp = """
		<div style="background-color:#44E3D3;padding:10px">
		<h1 style="color:white;text-align:center;font-family: 'Times New Roman', Times, serif;">Natural Language Processing</h1>
		</div>
		<br></br>
		<br></br>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

def display_pie_chart(df, column):
    counts = df[column].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title(f'Jumlah {column.capitalize()} dengan Masing-Masing Sentimen')
    st.pyplot(fig)
    

def front_down():
    #closing remarks
    pass



def contact():
    pass    
	#st.markdown(html,unsafe_allow_html=True)
