from pathlib import Path

import streamlit as st
from streamlit_extras.colored_header import colored_header

from src.setup.paths import IMAGES_DIR


colored_header(label=":violet[About Me]", description="", color_name="green-70")


col1, col2, col3 = st.columns([1,2,1])

with col2:
    profile_image_path: Path = IMAGES_DIR.joinpath("profile.jpeg")
    st.image(str(profile_image_path), width=300)

st.markdown(
    """
    Hi there! My name is Kobina, and I'm a mathematician and Machine Learning Engineer.
    Thanks for using the service. You can view its code [here](https://app.radicle.xyz/nodes/kobina.seednode.xyz/rad:zVhC4MGvgB8YMjvBuBNoQsSGtac6).

    Making this application was quite the adventure, and it's been a very rewarding experience that has taught me a great deal. 
    I'll continue to maintain and improve the system, and I hope that Lyft continue to provide data on a monthly basis. 

    I am currently building other machine learning systems, and incorporating elements of distributed systems into my skillset. So I 
    try to keep myself busy with these projects when I'm not at my day job.

    If you have data, and some ideas about how a AI/ML application can be built around it to bring value to your business, 
    consider reaching out to me.
    
    You can find a link to my publicly available work [here](https://app.radicle.xyz/nodes/kobina.seednode.xyz/rad:zVhC4MGvgB8YMjvBuBNoQsSGtac6), and my 
    LinkedIn [here](https://www.linkedin.com/in/kobina-brandon-aa9445134).
    """
)

