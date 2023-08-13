---
title: "Course Work"
permalink: /course_work/
search: true
---

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
        max-width: 1080px;
        margin: 50px auto;
        padding: 0 20px;
        position: relative;
        }
        .wrapper .center-line{
        position: absolute;
        height: 100%;
        width: 4px;
        background:  #000066; /* Change timeline color */
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
        background: #000066; /* Change box color */
        color: white;
        border-radius: 5px;
        width: calc(50% - 40px);
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
    </style>
</head>
<body>
 <p><strong>Purdue University - West Lafayette</strong></p>
 <p><strong>Bachelor of Science in Computer Science</strong></p>
    <ul>
        <li><p>Concentations in</p>
        <ul>
            <li><em>Software Engineering</em></li>
            <li><em>Machine Intelligence</em></li>
        </ul>
        </li>
        <li><p>Minor in Statistics </p></li>
    </ul>
    <p><strong>Bachelor of Science in Data Science</strong></p>
    <p><strong>Bachelor of Science in Artificial Intelligence</strong></p>
   <div class="wrapper" style="margin-left: 40px;"> <!-- Moved timeline to the right -->
    <div class="center-line">
      <a href="#" class="scroll-icon"><i class="fas fa-caret-up"></i></a>
    </div>
    <div class="row row-1">
      <section>
       <img src="/assets/image/nuvve.jpeg" alt="Image Description" class="icon">
        <div class="details">
          <span class="title">Fall 2023</span>
        </div>
         <ul class="sub-bullet">
            <li>Applied Regression Analysis</li>
            <li>Data Structure and Algorithm</li>
            <li>Statistical Theory</li>
            <li>Computer Architecture</li>
            <li>Data Mine Seminar III</li>
        </ul>
      </section>
    </div>
    <div class="row row-2">
      <section>
        <div class="details">
          <span class="title">Summer 2023</span>
        </div>
        <p>Probability<br>Elementary of Psychology</p>
      </section>
    </div>
    <div class="row row-1">
      <section>
        <i class="icon fas fa-rocket"></i>
        <div class="details">
          <span class="title">Spring 2023</span>
          <span>3rd Jan 2021</span>
        </div>
        <p>Data Engineering in Python<br>C Programming<br>Discrete Mathematics<br>Statistics for Data Science<br>Linear Algebra<br>Python Programming<br>Data Mine Seminar II</p>
      </section>
    </div>
    <div class="row row-2">
      <section>
        <div class="details">
          <span class="title">Fall 2022</span>
        </div>
        <p>Object Oriented Programming (JAVA)<br>Multivariable Calculus<br>Data Mine Seminar I</p>
      </section>
  </div>
