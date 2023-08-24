

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
}
.article {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 20px;
    width: 80%;
    max-width: 800px; /* Adjust the maximum width as needed */
    padding: 20px;
    box-shadow: 0 30px 60px -12px rgba(50, 50, 93, 0.25), 0 18px 36px -18px rgba(0, 0, 0, 0.3), 0 -12px 36px -8px rgba(0, 0, 0, 0.025);
    background: #fff;
    border-radius: 6px;
}
.thumbnail {
    max-width: 40%;
    height: auto;
    border-radius: 6px;
    object-fit: cover;
}
.article-info {
    flex: 1;
    padding: 20px;
}
.article-name {
    font-size: 1.2rem;
    line-height: 1.4;
    text-transform: uppercase;
    font-weight: 600;
    color: #9251ac;
    letter-spacing: 1.5px;
}
.article-comment {
    font-size: 1rem;
    line-height: 1.5;
}
.more {
    color: #9251ac;
    font-weight: bold;
    text-decoration: none;
}
</style>
  <div class="profile-section">
      <div class="introduction">
        <h2>ABOUT ME</h2>
          <p>"Hi, I'm <a class="sample">Aaron Kim </a>, an undergraduate pursuing a triple major in Computer Science with a focus on Machine Intelligence and Software Engineering, Data Science, and Artificial Intelligence. <br>
          I'm deeply passionate about AI, especially in Computer Vision and Natural Language Processing. I'm driven by curiosity about cutting-edge tech and its real-world use. <br>
          With a versatile skill set, I'm proficient in languages like C++, Python, Java, and more. I love coding and exploring innovative solutions through algorithms. <br>
          Currently, I'm expanding my knowledge in algorithms, machine learning, and AI. I excel in teamwork and seek chances to contribute to impactful projects with fellow professionals. </p>
          <h2> Language </h2>
          <img src="https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white"> <img src="https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Java-007396?style=flat-square&logo=java&logoColor=white"> <img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=Rust&logoColor=white"> <img src="https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white"> <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white">  <img src="https://img.shields.io/badge/Swift-F05138?style=flat-square&logo=swift&logoColor=white"> <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=SQL&logoColor=white"> <img src="https://img.shields.io/badge/Ruby-CC342D?style=flat-square&logo=Ruby&logoColor=white"> 
          <h2>Tools</h2>
           <p  id="capability"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"> <img src="https://img.shields.io/badge/Xcode-147EFB?style=flat-square&logo=xcode&logoColor=white"> <img src="https://img.shields.io/badge/PyCharm-000000?style=flat-square&logo=pycharm&logoColor=white">  <img src="https://img.shields.io/badge/IntelliJ IDEA-000000?style=flat-square&logo=intellijidea&logoColor="> <img src="https://img.shields.io/badge/Visual Studio Code-007ACC?style=flat-square&logo=visual studio code&logoColor=white"> <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/RStudio-75AADB?style=flat-square&logo=rstudio&logoColor=black"></p>
        </div>
    </div>
</html>
