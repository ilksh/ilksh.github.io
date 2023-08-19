---
title: "Aaron Kim"
permalink: /about/
excerpt: "About Me"
search: true
---

## Start 

<html lang="en">
<style>
body{
    background-color: #EFE7E3;
}
.frame{
    margin: 30px 150px;
    display: flex;
    flex-direction: column;
}
/*navigator*/
.navigator{
    display: flex;
    flex-direction: row;
    justify-content: space-between;
}
li{
    display: inline;
    align-self: center;
    margin:0;
    margin-left: 15px;
}
ul{
    margin:0;
}
.site-name{
    margin:0;
    font-family: 'Montserrat', sans-serif;
    font-size: 24px;
    color: #C2B4AB;
}
.nav-menu{
    font-family: 'Montserrat', sans-serif;
    font-weight: 400;
    font-size: 16px;
}
a{
    color: black;
    text-decoration: none;
}
a:hover{
    color: white;
    transition: 0.3s;
}
/*dashboard*/
.dashboard{
    display: flex;
    flex-direction: column;
    margin-top: 100px;
    margin-bottom: 70px;
}
.korean-comment{
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 500;
    font-size: 50px;
}
.sample{
    background-color: pink;
    -webkit-text-stroke: 1px black;
}
.sample:hover{
    background-color: transparent;
    color: white;
    -webkit-text-stroke: 1px black;
    transition: none;
}
.highlight{
    color: white;
    -webkit-text-stroke: 1px black;
}
.highlight:hover{
    background-color: pink;
    color: black;
    transition: none;
}
.type1:hover{
    background-color: yellowgreen;
}
.type2:hover{
    background-color: skyblue;
}
.type3:hover{
    background-color: orange;
}
.english-comment{
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 100;
    font-size: 15px;
    font-style: italic;
    letter-spacing: -0.3px;
}
/*recent works*/
.recent-menu{
    font-family: 'Montserrat', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #C2B4AB;
    text-align: right;
    margin-top: 100px;
}
/*footer*/
.footer{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 200px;
}
.contact{
    font-family: 'Montserrat', sans-serif;
    font-size: 50px;
    font-weight: 700;
    color: black;
    margin-bottom: 40px;
    border-bottom: 2px solid black;
}
.links>a{
    font-family: 'Montserrat', sans-serif;
    font-size: 30px;
    font-weight: 400;
    margin-left: 15px;
}
.footer>p{
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 100;
    font-size: 15px;
    font-style: italic;
    margin: 0;
    margin-top: 5px;
}
/* INFO Section*/
/*profile-section*/
.profile-section{
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: flex-start;
    margin: 100px 100px;
}
#profile-picture{
    width: 400px;
    margin-right: 60px;
}
.introduction>h2{
    margin: 0px;
    font-family: 'Montserrat', sans-serif;
    font-size: 35px;
}
.introduction>p{
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 300;
    font-size: 16px;
    margin-bottom: 40px;
}
#capability{
    font-family: 'Montserrat', sans-serif;
    font-weight: 400;
    font-size: 20px;
    font-style: italic;
    line-height: 30px;
}
/*value section*/
.value-menu{
    font-family: 'Montserrat', sans-serif;
    font-size: 35px;
    margin: 30px 100px;
}
.value-section{
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-content: center;
    margin: 100px 150px;
    margin-top: 20px;
}
.value{
    display: flex;
    flex-direction: row;
    margin-bottom: 40px;
    align-items: center;
}
.value-icon{
    width: 100px;
    height: 100px;
    border: 2px solid #C2B4AB;
    border-radius: 100px;
    padding: 15px;
}
.value-intro{
    margin-left: 30px;
}
.value-name{
    font-family: 'Montserrat', sans-serif;
    font-weight: 700px;
    margin-top: 0px;
    margin-bottom: 10px;
}
.value-exp{
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 300;
    font-size: 16px;
    margin: 0px;
}
/*footer section*/
.footer-comment{
    text-align: center;
}
a[title="Freepik"]:hover{
    color: black;
    text-decoration: underline;
}
a[title="Flaticon"]:hover{
    color: black;
    text-decoration: underline;
}
/* RECENT WORKS */
.recent-works{
    display: flex;
    flex-direction: column;
}
.article{
    display: flex;
    flex-direction: row;
    margin-bottom: 120px;
    align-items: center;
}
.thumbnail{
    width: 600px;
}
.article-info{
    margin: 0px 20px;
}
.article-name{
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 50px;
    margin-top: 0px;
    margin-bottom: 30px;
}
.article-name:hover{
    color: pink;
    -webkit-text-stroke: 1px black;
    transition: 0.3s;
}
.article-comment{
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 16px;
    font-weight: 100;
}
.more{
    width: content;
    height: content;
    float: right;
    margin-right: 10px;
    font-family: 'Montserrat', sans-serif;
    font-size: 14px;
    background-color: transparent;
    border: none;
    outline: none;
    padding: 0px;
}
#name1{
    position: relative;
    right: 70px;
}
#name1:hover{
    color: skyblue;
}
#name2-1{
    text-align: right;
    position: relative;
    left: 0px;
    margin-bottom: 0px;
}
#name2-2{
    text-align: right;
    position: relative;
    left: 100px;
}
#name2-2:hover{
    color: orange;
}
</style>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>INFO</title>

</head>
<body>
    <div class="frame">
        <nav class="navigator">
            <h3 class="site-name">ZOZELAND</h3>
            <ul>
                <li><a class="nav-menu" href="index.html">WORK</a></li>
                <li><a class="nav-menu" href="info.html">INFO</a></li>
            </ul>
        </nav>
        <div class="profile-section">
            <img id="profile-picture" src="profile.jpg">
            <div class="introduction">
                <h2>ABOUT ME</h2>
                <p>"기획, 디자인, 개발 다 하는 인간 스타트업이 되도록 노력 중"<br>산업공학과 출신, UX 관점에서 제대로 된 서비스 기획을 하기 위해 Front-End를 공부 중입니다. Adobe Tool들은 아직 잘 못 다루지만 디자인 감각은 있어요. :) 정보의 평등과 지식 공유의 중요성에 공감하고 있습니다. 무엇이든 시작하면 제대로 하는 성격! 같이 즐겁게, 열정적으로 진짜 뭔가를 만들어봐요! </p>
                <h2>CAPABILITY</h2>
                <p  id="capability">UX Research<br>UI Prototyping<br>Publishing with html/css</p>
            </div>
        </div>
        <h5 class="value-menu">MY VALUES</h5>
        <div class="value-section">
            <div class="value">
                <img class="value-icon" src="together.png">
                <div class="value-intro">
                    <h4 class="value-name">GO TOGETHER</h4>
                    <p class="value-exp">Lorem ipsum dolor sit amet consectetur adipisicing elit. Tempora sunt veritatis ea placeat. Iure, quam laudantium pariatur eligendi error eos voluptatum eaque maxime. Necessitatibus, nihil. Deleniti vitae perspiciatis est vel.</p>
                </div>
            </div>
            <div class="value">
                <img class="value-icon" src="partnership.png">
                <div class="value-intro">
                    <h4 class="value-name">RESPONSIBILITY</h4>
                    <p class="value-exp">Lorem ipsum dolor sit amet consectetur adipisicing elit. Fuga labore ad, perspiciatis ipsa veritatis neque et nostrum porro iste magnam dolorum similique laborum doloribus in possimus eveniet dicta voluptatum est!</p>
                </div>
            </div>
            <div class="value">
                <img class="value-icon" src="dove.png">
                <div class="value-intro">
                    <h4 class="value-name">COMMUNICATION</h4>
                    <p class="value-exp">Lorem ipsum dolor sit amet consectetur adipisicing elit. Quas ipsa iste blanditiis enim. Vel minus perferendis praesentium architecto deserunt, ipsam dolore hic, ullam nulla aut magnam eveniet facilis explicabo odio.</p>
                </div>
            </div>
        </div>
        <footer class="footer">
            <h3 class="contact">CONTACT ME</h3>
            <div class="links">
                <a class="blog" href="zozeland.tistory.com">BLOG</a>
                <a class="instagram" href="instagram.com/a_i_siteru">INSTAGRAM</a>
            </div>
            <p class="footer-comment"><br>Feel free to contact me!<br>Icons made by <a href="http://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a></p>
        </footer>
    </div>
</body>
</html>