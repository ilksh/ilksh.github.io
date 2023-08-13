---
title: "Experience"
permalink: /experience/
search: true
---

## Exp 1

<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Responsive Timeline Design | CodingNepal</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;300;400;500;600;700&display=swap');
        *{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: "Poppins", sans-serif;
        }
        html{
        scroll-behavior: smooth;
        }
        body{
        background:  #fff; /* Change background color to white */
        }
        ::selection{
        color: #fff;
        background: #ff7979;
        }
        .wrapper{
        max-width: 1080px; /* change width from 1080 to 1300*/
        margin: 50px auto;
        padding: 0 20px;
        position: relative;
        }
        .wrapper .center-line{
        position: absolute;
        height: 100%;
        width: 4px;
        background:  #000066; /* Change timeline color to "#b3ecff" */
        left: 50%;
        top: 20px;
        transform: translateX(-50%);
        }
        .wrapper .row{
        display: flex;
        }
        .wrapper .row-1{
        justify-content: flex-start;
        }
        .wrapper .row-2{
        justify-content: flex-end;
        }
        .wrapper .row section{
        background: #800000; /* Change box color */
        color: white;
        border-radius: 5px;
        width: calc(55% - 40px); /*change the width from 50% to 60% */
        padding: 20px;
        position: relative;
        }
        .wrapper .row section::before{
        position: absolute;
        content: "";
        height: 15px;
        width: 15px;
        background: #800000; /* Change box color to "#b3ecff" */
        top: 28px;
        z-index: -1;
        transform: rotate(45deg);
        }
        .row-1 section::before{
        right: -7px;
        }
        .row-2 section::before{
        left: -7px;
        }
        .row section .icon,
        .center-line .scroll-icon{
        position: absolute;
        background: #f2f2f2;
        height: 40px;
        width: 40px;
        text-align: center;
        line-height: 40px;
        border-radius: 50%;
        color: #ffffff; /* Change font color to white */
        font-size: 17px;
        box-shadow: 0 0 0 4px #fff, inset 0 2px 0 rgba(0,0,0,0.08), 0 3px 0 4px rgba(0,0,0,0.05);
        }
        .center-line .scroll-icon{
        bottom: 0px;
        left: 50%;
        font-size: 25px;
        transform: translateX(-50%);
        }
        .row-1 section .icon{
        top: 15px;
        right: -60px;
        }
        .row-2 section .icon{
        top: 15px;
        left: -60px;
        }
        .row section .details,
        .row section .bottom{
        display: flex;
        align-items: center;
        justify-content: space-between;
        }
        .row section .details .title{
        font-size: 22px;
        font-weight: 600;
        color: white; /* Change font color to white */
        }
        .row section p{
        margin: 10px 0 17px 0;
        }
        .row section .bottom a{
        text-decoration: none;
        background: #ff7979;
        color: #fff;
        padding: 7px 15px;
        border-radius: 5px;
        /* font-size: 17px; */
        font-weight: 400;
        transition: all 0.3s ease;
        }
        .row section .bottom a:hover{
        transform: scale(0.97);
        }
        @media(max-width: 790px){
        .wrapper .center-line{
            left: 40px;
        }
        .wrapper .row{
            margin: 30px 0 3px 60px;
        }
        .wrapper .row section{
            width: 100%;
        }
        .row-1 section::before{
            left: -7px;
        }
        .row-1 section .icon{
            left: -60px;
        }
        }
        @media(max-width: 440px){
        .wrapper .center-line,
        .row section::before,
        .row section .icon{
            display: none;
        }
        .wrapper .row{
            margin: 10px 0;
        }
        /* Change font color to white in the timeline */
        .row section {
            color: white;
        }
    }
     ul.sub-bullet {
            list-style-type: disc;
            margin-left: 20px;
        }
     .content-container {
    margin-left: 40px; /* Adjust this value as needed */
  }
    </style>
</head>
<body>
   <div class="wrapper" style="margin-left: 125px;"> <!-- Moved timeline to the right -->
    <div class="center-line">
      <a href="#" class="scroll-icon"><i class="fas fa-caret-up"></i></a>
    </div>
    <div class="row row-1">
      <section>
       <img src="/assets/image/helloworld.png" alt="Image Description" class="icon">
        <div class="details">
          <span class="title">Hackathon Mentor</span>
        </div>
        Purdue HelloWorld <br>
        <i> Sep 2023 </i>
         <ul class="sub-bullet">
            <li> Mentored participants in Communication team </li>
        </ul>
      </section>
    </div>
    <div class="row row-2">
      <section>
     <img src="/assets/image/PurdueUSB.png" alt="Image Description" class="icon">
        <div class="details">
          <span class="title">USB Tutor</span>
        </div>
        <i> Jan 2023 ~ May 2023 </i>
        <ul class="sub-bullet">
         <li>Tutored several Computer Science Courses</li>
         <li>Explained hard concepts</li>
        </ul>
      </section>
    </div>
    <div class="row row-1">
      <section>
          <img src="/assets/image/Nuvve.png" alt="Image Description" class="icon">
        <div class="details">
          <span class="title">DATA SCIENCE RESEARCH ASSISTANT</span>
        </div>
        Nuvve Holding Corp <br>
        <i> Aug 2022 ~ Dec 2022 </i>
        <ul class="sub-bullet">
        <li> Performed data cleaning, regression model training, and data visualizing </li>
        <li> Predicted driving patterns through clustering and Time-Series Analysis </li>
        </ul>
      </section>
    </div>
    <div class="row row-2">
      <section>
       <img src="/assets/image/mathTutor.png" alt="Image Description" class="icon">
        <div class="details">
          <span class="title">MATH & Coding TUTOR </span>
        </div>
       <i> Jan 2022 ~ June 2022 </i>
        <ul class="sub-bullet">
        <li>Created lectures, tests, and homework assignments</li>
        <li>Tutored Algorithms for problem solving</li>
        </ul>
      </section>
  </div>
