---
title: "Experience"
permalink: /experience/
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
      /* background: #87bbfe; */
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
      /* background: #ff9900; */
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
</style>
<div class="timeline">
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-1.5s timeline__event--type4">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Jan 2023 ~ <br> May 2023
		</div>
		<div class="timeline__event__content ">
    <div class="timeline__event__title">
				USB Tutor <br>
        Computer Science Undergraduate Board (USB)
			</div>
			<div class="timeline__event__description">
				<ul>
            <li>Provided individualized instruction in complex CS and MATH courses, specializing in Object Oriented Programming (OOP) and Multivariable Calculus.</li>
            <li> Elucidated intricate concepts while bolstering comprehension through illustrative examples, thereby facilitating a profound grasp of the subject matter. </li>
        </ul>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Aug 2022 ~ <br> Dec 2022
		</div>
		<div class="timeline__event__content">
    <div class="timeline__event__title">
				Data Science Research Assistant <br>
        Nuvve Holding Corp - Purdue Data Mine 
			</div>
			<div class="timeline__event__description">
				<ul>
        <li> Led regression analyses (linear and logistic) to reveal insights between variables while also applying k-means clustering to identify distinct driving patterns. </li>
        <li>Employed time series analysis to track evolving driving behaviors, informing data-driven decisions. </li> 
        </ul>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Jan 2022 ~ <br> Jun 2022
		</div>
		<div class="timeline__event__content">
    <div class="timeline__event__title">
			 Math & Coding Tutor
			</div>
			<div class="timeline__event__description">
				<ul>
        <li>Provided private math tutoring in SAT Math and AP Calculus with a comprehensive approach, including engaging lectures, challenging tests, and thought-provoking assignments. </li>
        <li> Also served as a coding tutor, teaching fundamental algorithms in C++ and Python, covering data structures, graph theory, shortest path, and segment tree concepts. </li>
        </ul>
			</div>
		</div>
	</div>
  <!--fourth-->
  <div class="timeline__event animated fadeInUp delay-1s timeline__event--type5">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Aug 2023 ~ <br> Current
		</div>
		<div class="timeline__event__content">
    <div class="timeline__event__title">
			 Hackathon Mentor <br> Purdue Hello World
			</div>
			<div class="timeline__event__description">
				<ul>
        <li> Serving as a mentor for freshman participants at a Purdue University hackathon, where I provided guidance and support to help them excel in their innovative projects. </li>
        </ul>
			</div>
		</div>
	</div>
</div>
</html>