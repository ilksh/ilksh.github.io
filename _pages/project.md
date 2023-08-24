---
title: "Aaron Kim"
permalink: /project/
excerpt: "Project"
search: true
---

<html>
<script src="https://kit.fontawesome.com/fc596df623.js" crossorigin="anonymous"></script>
<style>
  * {
    box-sizing: border-box;
    }
    html {
      font-size: 14px;
    }
    body {
      background: #f6f9fc;
      font-family: "Open Sans", sans-serif;
      color: #525f7f;
    }
    h2 {
      margin: 5%;
      text-align: center;
      font-size: 4rem;
      font-weight: 100;
    }
    h1 {
      margin: 4%;
      text-align: center;
      font-size: 2rem;
      font-weight: 10;
      top: 0;
    }
    .timeline {
      display: flex;
      flex-direction: column;
      margin: 20px auto;
      position: relative;
    }
    .timeline__event {
      margin-bottom: 20px;
      position: relative;
      display: flex;
      margin: 20px 0;
      border-radius: 6px;
      align-self: flex-end; /* Change this line to align the events to the right | Change from center */
      width: 50vw;
      margin-left: auto; /* Add this line to adjust the left margin */
    }
    .timeline__event:nth-child(2n+1) {
      flex-direction: row-reverse;
    }
    .timeline__event:nth-child(2n+1) .timeline__event__date {
      border-radius: 0 6px 6px 0;
    }
    .timeline__event:nth-child(2n+1) .timeline__event__content {
      border-radius: 6px 0 0 6px;
    }
    .timeline__event:nth-child(2n+1) .timeline__event__icon:before {
      content: "";
      width: 2px;
      height: 100%;
      background: #f6a4ec;
      position: absolute;
      top: 0%;
      left: 50%;
      right: auto;
      z-index: -1;
      transform: translateX(-50%);
      -webkit-animation: fillTop 2s forwards 4s ease-in-out;
              animation: fillTop 2s forwards 4s ease-in-out;
    }
    .timeline__event:nth-child(2n+1) .timeline__event__icon:after {
      content: "";
      width: 100%;
      height: 2px;
      background: #f6a4ec;
      position: absolute;
      right: 0;
      z-index: -1;
      top: 50%;
      left: auto;
      transform: translateY(-50%);
      -webkit-animation: fillLeft 2s forwards 4s ease-in-out;
              animation: fillLeft 2s forwards 4s ease-in-out;
    }
    .timeline__event__title {
      font-size: 1.2rem;
      line-height: 1.4;
      text-transform: uppercase;
      font-weight: 600;
      color: #9251ac; /* purple */
      letter-spacing: 1.5px;
    }
    .timeline__event__content {
      padding: 20px;
      box-shadow: 0 30px 60px -12px rgba(50, 50, 93, 0.25), 0 18px 36px -18px rgba(0, 0, 0, 0.3), 0 -12px 36px -8px rgba(0, 0, 0, 0.025);
      background: #fff;
      width: calc(40vw - 84px);
      border-radius: 0 6px 6px 0;
    }
    .timeline__event__date {
      color: #f6a4ec;
      font-size: 1.5rem;
      font-weight: 600;
      background: #9251ac;
      display: flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
      padding: 0 20px;
      border-radius: 6px 0 0 6px;
    }
    .timeline__event__icon {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #9251ac;
      padding: 20px;
      align-self: center;
      margin: 0 20px;
      background: #f6a4ec;
      border-radius: 100%;
      width: 40px;
      box-shadow: 0 30px 60px -12px rgba(50, 50, 93, 0.25), 0 18px 36px -18px rgba(0, 0, 0, 0.3), 0 -12px 36px -8px rgba(0, 0, 0, 0.025);
      padding: 40px;
      height: 40px;
      position: relative;
    }
    .timeline__event__icon i {
      font-size: 32px;
    }
    .timeline__event__icon:before {
      content: "";
      width: 2px;
      height: 100%;
      background: #f6a4ec;
      position: absolute;
      top: 0%;
      z-index: -1;
      left: 50%;
      transform: translateX(-50%);
      -webkit-animation: fillTop 2s forwards 4s ease-in-out;
              animation: fillTop 2s forwards 4s ease-in-out;
    }
    .timeline__event__icon:after {
      content: "";
      width: 100%;
      height: 2px;
      background: #f6a4ec;
      position: absolute;
      left: 0%;
      z-index: -1;
      top: 50%;
      transform: translateY(-50%);
      -webkit-animation: fillLeftOdd 2s forwards 4s ease-in-out;
              animation: fillLeftOdd 2s forwards 4s ease-in-out;
    }
    .timeline__event__description {
      flex-basis: 100%;
    }
    /* Math and Coding Tutor */
    .timeline__event--type2:after {
     /* background: #555ac0; */
      background: none;
    }
    .timeline__event--type2 .timeline__event__date {
      color: #87bbfe;
      background: #555ac0;
    }
    .timeline__event--type2:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type2:nth-child(2n+1) .timeline__event__icon:after {
      background: #87bbfe;
    }
    .timeline__event--type2 .timeline__event__icon {
      background: white;
      background-image: url('/assets/image/mathTutor.png');
       background-repeat: no-repeat;
      background-position: center center;
      background-size: contain;
      align-self: center;
      position: relative;
      border-radius: 100%; /* Make the icon circular */
      border: 2px solid black; /* Add a black border */
    }
    .timeline__event--type2 .timeline__event__icon:before, .timeline__event--type2 .timeline__event__icon:after {
      content: "";
      background: #87bbfe;
    }
    .timeline__event--type2 .timeline__event__title {
      color: #555ac0;
    }
    /* Nuvve */
    .timeline__event--type3:after {
      /* background: #24b47e; */
      background: none;
    }
    .timeline__event--type3 .timeline__event__date {
      color: #aff1b6; /* dark yellow */
      background-color: #24b47e;
    }
    .timeline__event--type3:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type3:nth-child(2n+1) .timeline__event__icon:after {
      background: #aff1b6;
    }
    .timeline__event--type3 .timeline__event__icon {
      background: white;
      background-image: url('/assets/image/Nuvve.png');
       background-repeat: no-repeat;
      background-position: center center;
      background-size: contain;
      align-self: center;
      position: relative;
      border-radius: 100%; /* Make the icon circular */
      border: 2px solid black; /* Add a black border */
    }
    .timeline__event--type3 .timeline__event__icon:before, .timeline__event--type3 .timeline__event__icon:after {
      content: "";
      background: #aff1b6; 
    }
    .timeline__event--type3 .timeline__event__title {
      color: #24b47e;
    }
    /* PURDUE USB */
     .timeline__event--type4:after {
      /* background: #ff9900; */
       background: none;
    }
    .timeline__event--type4 .timeline__event__date {
      color: #ffff4d;
      background: #ff9900;
    }
    .timeline__event--type4:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type4:nth-child(2n+1) .timeline__event__icon:after {
      background: #ffff4d; 
    }
    .timeline__event--type4 .timeline__event__icon {
      background: white;
       background-image: url('/assets/image/PurdueUSB.png');
       background-repeat: no-repeat;
      background-position: center center;
      background-size: contain;
      align-self: center;
      position: relative;
      border-radius: 100%; /* Make the icon circular */
      border: 2px solid black; /* Add a black border */
    }
    .timeline__event--type4 .timeline__event__icon:before, .timeline__event--type4 .timeline__event__icon:after {
      content: "";
      background: #ff9900;
    }
    .timeline__event--type4 .timeline__event__title {
      color: #ff9900;
    }
    /* hello world */ 
    .timeline__event--type5:after {
      /* background: #24b47e; */
      background: none;
    }
    .timeline__event--type5 .timeline__event__date {
      color: #9251ac;
      background-color: #f6a4ec;
    }
    .timeline__event--type5:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type5:nth-child(2n+1) .timeline__event__icon:after {
      background:#9251ac;
    }
    .timeline__event--type5 .timeline__event__icon {
      background: white;
      background-image: url('/assets/image/helloworld.png');
       background-repeat: no-repeat;
      background-position: center center;
      background-size: contain;
      align-self: center;
      position: relative;
      border-radius: 100%; /* Make the icon circular */
      border: 2px solid black; /* Add a black border */
    }
    .timeline__event--type5 .timeline__event__icon:before, .timeline__event--type5 .timeline__event__icon:after {
      content: "";
      background: #9251ac; 
    }
    .timeline__event--type5 .timeline__event__title {
      color: #f6a4ec;
    }
    .timeline__event:last-child .timeline__event__icon:before {
      content: none;
    }
    @media (max-width: 786px) {
      .timeline__event {
        flex-direction: column;
        align-self: center;
      }
      .timeline__event__content {
        width: 100%;
      }
      .timeline__event__icon {
        border-radius: 6px 6px 0 0;
        width: 100%;
        margin: 0;
        box-shadow: none;
      }
      .timeline__event__icon:before, .timeline__event__icon:after {
        display: none;
      }
      .timeline__event__date {
        border-radius: 0;
        padding: 20px;
      }
      .timeline__event:nth-child(2n+1) {
        flex-direction: column;
        align-self: center;
      }
      .timeline__event:nth-child(2n+1) .timeline__event__date {
        border-radius: 0;
        padding: 20px;
      }
      .timeline__event:nth-child(2n+1) .timeline__event__icon {
        border-radius: 6px 6px 0 0;
        margin: 0;
      }
    }
    @-webkit-keyframes fillLeft {
      100% {
        right: 100%;
      }
    }
    @keyframes fillLeft {
      100% {
        right: 100%;
      }
    }
    @-webkit-keyframes fillTop {
      100% {
        top: 100%;
      }
    }
    @keyframes fillTop {
      100% {
        top: 100%;
      }
    }
    @-webkit-keyframes fillLeftOdd {
      100% {
        left: 100%;
      }
    }
    @keyframes fillLeftOdd {
      100% {
        left: 100%;
      }
    }
    .recent-menu{
    font-family: 'Montserrat', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #C2B4AB;   
    text-align: right;
    margin-top: 100px;
}
.profile-section {
    display: flex;
    align-items: center;
    justify-content: flex-end; /* Align content to the right */
    margin-top: 50px;
}
#profile-picture {
    width: 150px; /* Adjust the width to make it square */
    height: 150px; /* Set the same height as width */
    border-radius: 50%;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    object-fit: cover; /* Maintain aspect ratio and cover the container */
}
.introduction {
    flex: 1;
    padding: 20px;
    text-align: left; /* Align text to the left */
}
.introduction h2 {
    margin-top: 0;
    font-size: 2rem;
}
.introduction p {
    font-size: 1rem;
    line-height: 1.5;
}
#capability {
    font-weight: bold;
}.recent-works {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 50px;
}    /* Adjustments for the image and text layout */
    .article {
      display: flex; /* Use flexbox to align items */
      align-items: center; /* Align items vertically */
      margin: 20px;
      width: 100%;
      max-width: 100%;
      padding: 20px;
      box-shadow: /* ... (box shadow styles) ... */;
      background: #fff;
      border-radius: 6px;
    }
    .thumbnail {
      flex: 1;
      max-width: 40%;
      max-height: 100%;
      border-radius: 6px;
      object-fit: cover;
    }
    .article-info {
      flex: 2;
      padding: 20px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    /* Other styles remain unchanged */
    @import url('https://fonts.googleapis.com/css?family=Poppins&display=swap');
@font-face {
    font-family: CircularStd;
    src: url(assets/fonts/CircularStd-Bold.eot);
    src: url("assets/fonts/CircularStd-Bold.eot?#iefix") format("embedded-opentype"), url("assets/fonts/CircularStd-Bold.woff") format("woff"), url("assets/fonts/CircularStd-Bold.ttf") format("truetype"), url("assets/fonts/CircularStd-Bold.svg#bcc26993292869431e54c666aafa8fcd") format("svg");
}
:root {
    --primary-color: none;
    --overlay-color: rgba(255, 255, 255, 0.9);
    --menu-speed: 0.75s;
    --light-text-color: #444;
    --black-color: #000;
    --white-color: #fff;
    --hero-color: black;
}
.work-hero-card{
    display: block;
    margin:0 auto;
    width:80%;
    background: #fff;
    align-items: center;
    padding:0;
    /* border-radius: 10px; */
    padding-bottom: 15px;
    /* border:1px solid #44444450; */
    box-shadow: 1px 1px 20px #00000025;
    cursor: pointer;
    transition: all 1s;
    height: 100%;
}
#work-text{
    text-align: center;
    font-size: 1.5em;
}
.work-hero-card h1{
    padding-top: 20px;
    text-align: center;
    font-size: 0.8em;
    font-weight: bold;
}
.work-hero-card:hover{
    width:85%;
    box-shadow: 0px 0px 20px #88888875;
    animation:zoom 1s;
    /* filter: none; */
}
.work-card-hero-img{
    width:100%;
    display: block;
    /* border:1px solid #44444450; */
    /* border-radius: 10px; */
    border-bottom: none;
    margin:0 auto;
    left:0;
    right: 0;
}
.work-double-row{
    padding-left:120px;
    padding-right: 120px;
    margin-top: 80px;
}
.project-star {
    position: absolute;
    background: #ffff0080;
    border-radius: 50px;
    padding: 2px 6px;
    font-size: 0.8em;
    top: 40px;
    right: 0;
}
.project-star-mobile {
    display: none;
    background: #ffff0080;
    border-radius: 50px;
    padding: 2px 6px;
    width: 50%;
    margin-bottom: 10px;
    text-align: center;
    font-size: 0.8em;
}
#work-row {
    padding-left: 135px;
    padding-right: 135px;
    width: 100vw;
}
.work-hero {
    /* border:2px solid #444; */
    border-radius: 10px;
    /* box-shadow: 0px 0px 5px #444; */
    width: 90%;
    transition: width 1s;
}
.work-hero:hover{
    width: 95%;
}
.work-des {
    padding-top: 40px;
}
.work-des h1 {
    font-weight: bold;
}
.work-des p {
    text-align: justify;
    font-weight: 100;
}
.work-hero-img {
    width: 100%;
    border-radius: 10px;
}
.technologies h2 {
    font-size: 0.8em;
    font-weight: bold;
}
.tech-stack {
    cursor: pointer;
    margin-right: 10px;
    background: #f2f2f2;
    padding: 2px 4px;
    font-size: 0.8em;
    transition: background 1s;
}
.tech-stack:hover {
    background: beige;
}
.featured {
    margin-top: 10px;
}
.partners {
    /* background:#f2f2f2; */
    height: 50px;
    margin-right: 20px;
}
#partner-logo {
    height: 40px;
    cursor: pointer;
    transition: all 1s;
}
#partner-logo:hover {
    box-shadow: 0px 0px 10px #444;
}
.work-read {
    float: right;
    font-size: 0.8em;
}
.work-read:hover {
    background: #444;
    color: white !important;
    border: 2px solid #444 !important;
}
body {
    font-family: 'Poppins', sans-serif;
    color: var(--light-text-color);
    overflow-x: hidden;
}
.comingsoon h1 {
    font-size: 3em;
}
.comingsoon {
    position: absolute;
    margin: auto;
    top: 0;
    /* background: white; */
    text-align: center;
    right: 0;
    bottom: 0;
    left: 0;
    /* width: 40em; */
    height: 10em;
    animation: opac 0.1s ease-in-out 5;
}
.comingsoon a {
    color: #444;
    font-weight: bold;
    text-decoration: none;
    transition: font-size 1s;
}
.comingsoon a:hover {
    color: blue;
    font-size: 1.5em;
}
.social-icons a {
    color: var(--light-text-color);
    text-decoration: none;
    transition: all 1s;
}
.social-icons a:hover {
    color: blue;
}
li span a {
    color: #444;
    text-decoration: none;
}
li span a:hover {
    color: #444;
    text-decoration: none;
}
.copyright-legal {
    font-size: 1.1em;
    color: #444;
    letter-spacing: 0px;
    text-align: left;
}
.legal-links {
    text-decoration: none;
    font-weight: bold;
    transition: all 1s;
}
.legal-links:hover {
    text-decoration: none;
    color: blue;
}
.legal a,
.legal a:hover,
.legal a:active {
    color: #444;
    text-decoration: none;
}
.privacy-policy h1 {
    font-size: 2.5em;
    /* font-family: 'CircularStd'; */
    font-weight: bold;
    letter-spacing: 0.1rem;
}
.privacy-policy h2 {
    font-size: 25px;
    font-family: 'CircularStd';
    /* letter-spacing: 0.1rem; */
}
.legal-des {
    text-align: justify;
    font-weight: 500;
}
.legal-des strong {
    /* font-size: 1.1em; */
    font-weight: 900;
}
.privacy-policy p {
    padding-right: 13vw;
    letter-spacing: 0.8px;
}
.nav-desk {
    height: 10vh;
    /* position: fixed; */
    display: block;
    width: 80vw;
    margin: auto;
    margin-top: 20px;
    left: 0;
    right: 0;
    /* background: black; */
}
.nav-logo {
    height: 100%;
    float: left;
}
.menu-wrap {
    position: fixed;
    top: 50px;
    right: 50px;
    z-index: 1;
    display: none;
}
.menu-wrap .toggler {
    position: absolute;
    top: 0px;
    right: 0;
    z-index: 2;
    cursor: pointer;
    width: 50px;
    height: 50px;
    opacity: 0;
}
.menu-wrap .hamburger {
    position: absolute;
    top: 0;
    right: 0;
    z-index: 1;
    width: 60px;
    height: 60px;
    padding: 1rem;
    background: var(--primary-color);
    display: flex;
    align-items: center;
    justify-content: center;
}
/* Hamburger Line */
.menu-wrap .hamburger>div {
    position: relative;
    flex: none;
    width: 100%;
    height: 2px;
    z-index: 1;
    background: black;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.4s ease;
}
/* Hamburger Lines - Top & Bottom */
.menu-wrap .hamburger>div::before,
.menu-wrap .hamburger>div::after {
    content: '';
    position: absolute;
    z-index: 1;
    top: -10px;
    width: 100%;
    height: 2px;
    background: inherit;
}
/* Moves Line Down */
.menu-wrap .hamburger>div::after {
    top: 10px;
}
/* Toggler Animation */
.menu-wrap .toggler:checked+.hamburger>div {
    transform: rotate(135deg);
}
/* Turns Lines Into X */
.menu-wrap .toggler:checked+.hamburger>div:before,
.menu-wrap .toggler:checked+.hamburger>div:after {
    top: 0;
    transform: rotate(90deg);
}
/* Rotate On Hover When Checked */
.menu-wrap .toggler:checked:hover+.hamburger>div {
    transform: rotate(225deg);
}
/* Show Menu */
.menu-wrap .toggler:checked~.menu {
    visibility: visible;
}
.menu-wrap .toggler:checked~.menu>div {
    transform: scale(1);
    transition-duration: var(--menu-speed);
}
.menu-wrap .toggler:checked~.menu>div>div {
    opacity: 1;
    transition: opacity 0.4s ease 0.4s;
}
.menu-wrap .menu {
    position: fixed;
    top: 0;
    right: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    visibility: hidden;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}
.menu-wrap .menu>div {
    background: var(--overlay-color);
    border-radius: 50%;
    width: 200vw;
    height: 200vw;
    display: flex;
    flex: none;
    align-items: center;
    justify-content: center;
    transform: scale(0);
    transition: all 0.4s ease;
}
.menu-wrap .menu>div>div {
    text-align: center;
    max-width: 90vw;
    max-height: 100vh;
    opacity: 0;
    transition: opacity 0.4s ease;
}
.menu-wrap .menu>div>div>ul>li {
    list-style: none;
    color: var(--white-color);
    font-size: 1.5rem;
    padding: 1rem;
}
.menu-wrap .menu>div>div>ul>li>a {
    color: inherit;
    text-decoration: none;
    transition: color 0.4s ease;
}
.legal {
    position: absolute;
    bottom: 25px;
    right: 25px;
}
.hero-tag {
    margin-top: 20vh;
    text-align: left;
    font-size: 4.5em;
    font-weight: bold;
    color: var(--hero-color);
    font-family: 'CircularStd', sans-serif;
}
.contact-hero {
    margin-top: 0vh;
    font-size: 3.5em;
}
.hero-des {
    padding-right: 60px;
    text-align: justify;
    /* font-weight: 800; */
    letter-spacing: 1.2;
}
.contact-des {
    letter-spacing: 0.2;
}
.input-container {
    padding: 10px 0px;
}
.form-group {
    margin-bottom: 1rem;
}
label {
    display: inline-block;
    margin-bottom: 0.5rem;
}
.form-control {
    height: calc(2.25rem + 2px);
    padding: .5rem .75rem;
    width: 90%;
    font-size: 1rem;
    line-height: 1.5;
    color: #495057;
    background: none;
    border: none;
    border-bottom: 2px solid #444;
    border-radius: 0rem;
    transition: all .5s ease-in-out;
    outline: none;
}
.form-control:focus {
    outline: none;
    border-bottom: 2px solid blue;
    box-shadow: none;
}
.form-control:hover {
    background: whitesmoke;
}
/* input:focus, textarea:focus, select:focus{
    outline: none;
} */
.learn-btn {
    padding: 5px 10px;
    background: var(--light-text-color);
    color: white;
    border: 2px solid var(--light-text-color);
    letter-spacing: 0.1;
    transition: all 1s;
}
.contact-des a {
    text-decoration: none;
    color: #444;
    transition: color 1s;
}
.contact-des a:hover {
    color: blue;
}
.learn-btn:hover {
    background: blue;
    border: 2px solid blue;
}
.contact-btn {
    margin-left: 6px;
    padding: 5px 10px;
    color: var(--light-text-color);
    background: none;
    border: 2px solid var(--light-text-color);
    transition: all 1s;
}
.contact-btn:hover {
    color: blue;
    border: 2px solid blue;
}
#hero-img {
    margin-top: 50px;
    width: 90%;
}
#left-hero {
    padding-left: 150px;
}
#logo {
    margin-top: 5px;
    height: 80%;
    padding: 6px;
    border: 4px solid #444;
    border-radius: 50%;
}
/* .nav-items{
    color:black;
    float:right;
} */
ul {
    float: right;
    /* color:white; */
    list-style: none;
    padding: 30px 0px;
}
li {
    /* text-transform: uppercase; */
    display: inline;
    font-size: 16px;
    letter-spacing: 0.1;
    padding: 0px 15px;
}
.social-icons {
    margin-top: 30%;
}
.footer-full {
    width: 80vw;
    margin: 0 auto;
    padding: 30px 10px;
    padding-top: 100px;
    position: relative;
    bottom: 0;
    left: 0;
    right: 0;
}
.footer-credits {
    float: right;
    /* display: inline; */
}
.footer-credits a {
    color: #444;
    font-weight: bold;
    text-decoration: none;
    transition: all 1s;
}
.footer-credits a:hover {
    color: blue;
    text-decoration: none;
    font-weight: bold;
}
.social-icons i {
    font-size: 2em;
    /* font-weight: 100; */
    padding: 0px 5px;
}
li span {
    position: relative;
    /* display: block; */
    cursor: pointer;
}
li span:before,
li span:after {
    content: '';
    position: absolute;
    width: 0%;
    height: 1px;
    top: 50%;
    margin-top: -0.5px;
    background: var(--black-color);
}
li span:before {
    left: -2.5px;
}
li span:after {
    right: 2.5px;
    background: var(--black-color);
    transition: width 0.8s cubic-bezier(0.22, 0.61, 0.36, 1);
}
.footer-full {
    padding-top: 60px;
}
li span:hover:before {
    background: var(--black-color);
    width: 100%;
    transition: width 0.5s cubic-bezier(0.22, 0.61, 0.36, 1);
}
li span:hover:after {
    background: transparent;
    width: 100%;
    transition: 0s;
}
#index-body {
    /* background: url('assets/img/mobile.png'); */
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}
/* Tablets */
@media (min-width: 768px) and (max-width: 1024px) {
    #left-hero {
        padding-left: 80px;
        flex: 0 0 100% !important;
        max-width: 100% !important;
    }
    #work-row {
        padding-left: 50px;
        padding-right: 50px;
    }
    .work-hero {
        width: 100%;
    }
    #hero-right {
        display: none;
    }
    .work-double-row{
        padding-left: 30px;
        padding-right: 30px;
    }
    #index-body {
/*         background: url('assets/img/mobile.png'); */
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100vh;
        width: 100vw;
    }
    /* .work-double-row{
        padding-left: 50px;
        padding-right: 50px;
    } */
}
/* #logo {
    display: none;
} */
@media only screen and (max-width: 768px) {
    /* For mobile phones: */
    .nav-items {
        width: 0;
        height: 0;
        display: none;
    }
    .work-double-row{
        padding:30px 20px;
        padding-bottom: 0;
    }
    .work-hero-card{
        width:100%;
        margin-bottom: 25px;
    }
    #row-3{
        margin-top: 0px;
        padding:0 20px;
    }
    #index-body {
/*         background: url('assets/img/mobile.png'); */
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100vh;
    }
    #left-hero {
        padding-left: 50px;
        ;
    }
    #work-row {
        padding-right: 0;
        padding-left: 30px;
    }
    .work-des h1 {
        text-align: center;
    }
    .work-read {
        float: none;
    }
    #logo {
        display: none;
        /* height: 65%;
        margin-top: 28px; */
    }
    .hero-tag {
        margin-top: 20vh;
        font-size: 4em;
    }
    #hero-right {
        display: none;
    }
    .hero-des {
        padding-right: 30px;
    }
    .menu-wrap {
        display: block;
    }
    .menu ul li {
        display: block !important;
    }
    .menu ul li a {
        color: var(--black-color) !important;
    }
    .menu-wrap .hamburger {
        z-index: 2;
    }
    .menu-wrap .toggler {
        z-index: 3;
    }
    .contact-hero {
        margin-top: 10vh;
    }
    .footer-full {
        text-align: center;
        padding-top: 40px !important;
    }
    #contact-social {
        margin-top: 10%;
    }
    .footer-credits {
        display: block;
        text-align: center;
        padding-top: 10px;
        float: none;
    }
    #legal-hero {
        margin-top: 5vh !important;
        text-align: left;
    }
    #legal-hero-des {
        padding-right: 10px;
        text-align: left;
        font-size: 0.8em;
    }
    .privacy-policy p {
        padding-right: 1vw;
        /* text-align: left; */
    }
    .project-star {
        display: none;
    }
    .project-star-mobile {
        display: block;
    }
}
::-webkit-scrollbar {
    width: 5px;
}
/* Track */
::-webkit-scrollbar-track {
    background: white;
}
/* Handle */
::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 50px;
}
/* Handle on hover */
::-webkit-scrollbar-thumb:hover {
    background: #555; 
}
  </style>
<head>
    <script async src="https://www.googletagmanager.com/gtag/js?id=UA-159411627-1"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag() {
            dataLayer.push(arguments);
        }
        gtag('js', new Date());
        gtag('config', 'UA-159411627-1');
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.2/animate.min.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
    <link rel="stylesheet" href="style.css">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, shrink-to-fit=yes">
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css"
        integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
    <meta name="Description"
        content="Ishan Sharma ishandeveloper Ishan Inc. Chandigarh, India www.ishandeveloper.com .NET | ML | AI | Python | UI/UX | App Developer | Flutter | Web Developer" />
    <title>Work | Ishan Sharma</title>
    <link rel="icon" type="image/png" href="assets/img/favicon.png">
<body id="contact-body" style="overflow-x: hidden;">
    <div>
                <div class="nav-logo">
                    <a href="index.html"><img id="logo" src="assets/img/logo2.png"></a>
                </div>
                <div class="nav-items">
                    <ul>
                        <li><span><a href="about.html">About</a></span></li>
                        <li><span><a href="#" style="text-decoration: line-through;">Work</a></span></li>
                        <li><span><a href="uses.html">Uses</a></span></li>
                        <li><span><a href="contact.html">Contact</a></span></li>
                    </ul>
                </div>
        <div class=" fadeIn row" style="width:100vw;height:auto;animation-duration: 1.5s;">
            <div class="col-md-12" id="left-hero">
                <h1 class="hero-tag contact-hero  fadeInLeft">My Work.</h1>
                <p class="hero-des contact-des  fadeIn delay-1s">Take a look at some of my projects I've done,
                    apps or websites that I've made and my designs.
                </p>
                <!-- <div class="feedback"><a href="https://d474b570.nolt.io/">Have any feedback or suggestion?</a></div> -->
            </div>
        </div>
        <!--Content-->
        <div class="animated bounceInUp delay-1s">
            <div class="row" id="work-row">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-2s">
                        <div class="work-hero">
                            <img class="work-hero-img" src="/assets/image/MarketplaceDES.png">
                        </div>
                </div>
                <div class="col-md-6 col-sm-12">
                    <div class="work-des ">
                        <h1>Hotspoter</h1>
                            <div class="project-star-mobile  animated zoomIn delay-2s">
                                <i class="fas fa-trophy"></i> <b>5 Million+</b> Downloads
                            </div>
                        <p>Hotspoter is special software that allows users to transform their computer into a wireless
                            router. This is ideal for people who are looking for a way to use their existing internet
                            connection to surf on their Smartphone, while the connection can also be shared with a large
                            number of different people at the same time without weakening the signal strength.</p>
                        <div class="technologies">
                            <h2>Technologies Used</h2>
                            <span class="tech-stack">
                                CSharp
                            </span>
                            <span class="tech-stack">
                                .Net Framework
                            </span>
                        </div>
                        <div class="technologies featured">
                            <h2>Featured On</h2>
                            <span class="partners">
                                <a target="_blank" href="https://hotspoter.en.softonic.com"><img id="partner-logo"
                                        src="assets/img/work/hotspoter/softonic.png" style="padding:10px;"></a>
                            </span>
                            <span class="partners">
                                <a target="_blank"
                                    href="https://download.cnet.com/Hotspoter/3000-18508_4-76462434.html"><img
                                        id="partner-logo" src="assets/img/work/hotspoter/cnet.png"
                                        style="border-radius: 50%;"></a>
                            </span>
                        </div>
                        <!-- <button class="contact-btn work-read" style="margin:0;">
                            Read More
                        </button> -->
                    </div>
                    <div class="project-star animated zoomIn delay-2s">
                        <i class="fas fa-trophy"></i> <b>5 Million+</b> Downloads
                    </div>
                </div>
            </div>
            <div class="row work-double-row">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="/assets/image/MarketplaceDES.png">
                        <h1>mehaksharma.co</h1>
                        <div id="work-text">PORTFOLIO</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="/assets/image/MarketplaceDES.png">
                        <h1>ishandeveloper.com</h1>
                        <div id="work-text">PORTFOLIO</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="/assets/image/MarketplaceDES.png">
                        <h1>URL Shortener</h1>
                        <div id="work-text">WEB APP</div>                        
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="/assets/image/MarketplaceDES.png">
                        <h1>News App  [GoLang]</h1>
                        <div id="work-text">WEB APP</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/foodfuel/food.png">
                        <h1>FoodFuel</h1>
                        <div id="work-text">WEB APP</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/timestamp/hero.png">
                        <h1>Timestamp Microservice</h1>
                        <div id="work-text">API</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/todo/hero.png">
                        <h1>Task-To-Do</h1>
                        <div id="work-text">WEB APP</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/basketball/hero.png">
                        <h1>Basketball Tournament</h1>
                        <div id="work-text">WEB TEMPLATE</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/dictionary/hero.png">
                        <h1>Dictionary</h1>
                        <div id="work-text">FLUTTER APP</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/pokemon/hero.png">
                        <h1>PokemonPedia</h1>
                        <div id="work-text">FLUTTER APP</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/calculator/hero.png">
                        <h1>Calculator</h1>
                        <div id="work-text">FLUTTER APP</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/qr/hero.png">
                        <h1>QR Code Scanner</h1>
                        <div id="work-text">FLUTTER APP</div>
                    </div>
                </div>
            </div>
            <div class="row work-double-row" id="row-3">
                <div class="col-md-6 col-sm-12  bounceInLeft delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/ppt/2.png">
                        <h1>Presentations in Browser</h1>
                        <div id="work-text">WEB APP</div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12  bounceInRight delay-3s">
                    <div class="work-hero-card">
                        <img class="work-card-hero-img" src="assets/img/work/food/hero.png">
                        <h1>SeeFood</h1>
                        <div id="work-text">UI/UX</div>
                    </div>
                </div>
            </div>
        </div>
            <div class="footer-full  fadeIn delay-2s">
                <div id="contact-social" class="social-icons" style="display: inline;">
                    <a target="_blank" href="mailto:ishandeveloper@outlook.com"><i class="fas fa-envelope"></i></a>
                    <a target="_blank" href="https://instagram.com/developer.ishan"><i class="fab fa-instagram"></i></a>
                    <a target="_blank" href="https://hackerrank.com/ishandeveloper"><i
                            class="fab fa-hackerrank"></i></a>
                    <a target="_blank" href="https://github.com/ishandeveloper"><i class="fab fa-github"></i></a>
                    <a target="_blank" href="https://linkedin.com/in/ishandeveloper"><i class="fab fa-linkedin"></i></a>
                </div>
                <div class="footer-credits">
                    Made with <span style="color:red">♥</span> by <a
                        href="https://github.com/ishandeveloper">ishandeveloper</a>
                </div>
            </div>
        </div>
        <div class="menu-wrap">
            <input type="checkbox" class="toggler">
            <div class="hamburger">
                <div></div>
            </div>
            <div class="menu">
                <div>
                    <div>
                        <ul>
                            <li><a href="index.html">Home</a></li>
                            <li><a href="#">About</a></li>
                            <li><a href="#">Portfolio</a></li>
                            <li><a href="#">Uses</a></li>
                            <li><a disabled href="#" style="text-decoration: line-through;">Contact</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
</body>
</html>