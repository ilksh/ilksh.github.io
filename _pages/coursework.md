---
title: "Course Work"
permalink: /course_work/
search: true
---

# Course Work

<html lang="en">
<head>
  <title> Vertical Timeline </title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" type="text/css" href="css/style.css">
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    *{
        margin:0;
        padding:0;
        box-sizing: border-box;
    }
    body{
        font-family: 'Poppins', sans-serif;
    }
    .timeline-section{
        background-color: #24292d;
        min-height: 100vh;
        padding: 100px 15px;
    }
    .timeline-items{
        max-width: 1000px;
        margin:auto;
        display: flex;
        flex-wrap: wrap;
        position: relative;
    }
    .timeline-items::before{
        content: '';
        position: absolute;
        width: 2px;
        height: 100%;
        background-color: #2f363e;
        left: calc(50% - 1px);
    }
    .timeline-item{
        margin-bottom: 40px;
        width: 100%;
        position: relative;
    }
    .timeline-item:last-child{
        margin-bottom: 0;
    }
    .timeline-item:nth-child(odd){
        padding-right: calc(50% + 30px);
        text-align: right;
    }
    .timeline-item:nth-child(even){
        padding-left: calc(50% + 30px);
    }
    .timeline-dot{
        height: 16px;
        width: 16px;
        background-color: #eaa023;
        position: absolute;
        left: calc(50% - 8px);
        border-radius: 50%;
        top:10px;
    }
    .timeline-date{
        font-size: 18px;
        color: #eaa023;
        margin:6px 0 15px;
    }
    .timeline-content{
        background-color: #2f363e;
        padding: 30px;
        border-radius: 5px;
    }
    .timeline-content h3{
        font-size: 20px;
        color: #ffffff;
        margin:0 0 10px;
        text-transform: capitalize;
        font-weight: 500;
    }
    .timeline-content p{
        color: #c8c8c8;
        font-size: 16px;
        font-weight: 300;
        line-height: 22px;
    }

    /* responsive */
    @media(max-width: 767px){
        .timeline-items::before{
            left: 7px;
        }
        .timeline-item:nth-child(odd){
            padding-right: 0;
            text-align: left;
        }
        .timeline-item:nth-child(odd),
        .timeline-item:nth-child(even){
            padding-left: 37px;
        }
        .timeline-dot{
            left:0;
        }
    }
    <style>
</head>
<body>

<section class="timeline-section">
	<div class="timeline-items">
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2015</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2016</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2017</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2018</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2019</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2020</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
		<div class="timeline-item">
			<div class="timeline-dot"></div>
			<div class="timeline-date">2021</div>
			<div class="timeline-content">
				<h3>timeline item title</h3>
				<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>
			</div>
		</div>
	</div>
</section>

</body>
</html>

