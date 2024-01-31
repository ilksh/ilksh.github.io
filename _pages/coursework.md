---
title: "Course Work"
permalink: /course_work/
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
    .timeline__event--type2:after {
      background: #555ac0;
    }
    .timeline__event--type2 .timeline__event__date {
      color: #87bbfe;
      background: #555ac0;
    }
    .timeline__event--type2:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type2:nth-child(2n+1) .timeline__event__icon:after {
      background: #87bbfe;
    }
    .timeline__event--type2 .timeline__event__icon {
      background: #87bbfe;
      color: #555ac0;
    }
    .timeline__event--type2 .timeline__event__icon:before, .timeline__event--type2 .timeline__event__icon:after {
      background: #87bbfe;
    }
    .timeline__event--type2 .timeline__event__title {
      color: #555ac0;
    }
    .timeline__event--type3:after {
      background: #24b47e;
    }
    .timeline__event--type3 .timeline__event__date {
      color: #aff1b6;
      background-color: #24b47e;
    }
    .timeline__event--type3:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type3:nth-child(2n+1) .timeline__event__icon:after {
      background: #aff1b6;
    }
    .timeline__event--type3 .timeline__event__icon {
      background: #aff1b6;
      color: #24b47e;
    }
    .timeline__event--type3 .timeline__event__icon:before, .timeline__event--type3 .timeline__event__icon:after {
      background: #aff1b6;
    }
    .timeline__event--type3 .timeline__event__title {
      color: #24b47e;
    }
     .timeline__event--type4:after {
      background: #ff9900;
    }
    .timeline__event--type4 .timeline__event__date {
      color: #ffff4d;
      background: #ff9900;
    }
    .timeline__event--type4:nth-child(2n+1) .timeline__event__icon:before, .timeline__event--type4:nth-child(2n+1) .timeline__event__icon:after {
      background: #ffff4d;
    }
    .timeline__event--type4 .timeline__event__icon {
      background: #ff9900;
      color: #ffff4d;
    }
    .timeline__event--type4 .timeline__event__icon:before, .timeline__event--type4 .timeline__event__icon:after {
      background: #ff9900;
    }
     .timeline__event--type4 .timeline__event__title {
      color: #ffff4d;
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
<p><strong>Purdue University - West Lafayette</strong></p> 
  <ul>
    <li> Bachelor of Computer Science </li>
    <ul class="sub-bullet">
    <li> Concentrations in  <br> <i>Machine Intelligence</i> <br> <i>Algorithmic Foundations</i></li>
    <li> Minor in <i>Statistics</i> </li>
    </ul>
    <li>Bachelor of Science in Artificial Intelligence</li>
    <li>Bachelor of Science in Data Science</li>
  </ul>
<div class="timeline">
	<div class="timeline__event  animated fadeInUp delay-1s timeline__event--type2">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Fall 2023 
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__description">
				<ul>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=MA&term=CURRENT&cnbr=36200" target="_blank"> Vector Calculus </a> </li>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=TDM&term=CURRENT&cnbr=20200" target = "_blank"> Data Mine Seminar IV </a> </li>
        </ul>
			</div>
		</div>
	</div>
	<!--Fall 2023-->
	<div class="timeline__event  animated fadeInUp delay-1s timeline__event--type4">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Fall 2023 
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__description">
				<ul>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=STAT&term=CURRENT&cnbr=51200" target="_blank"> Applied Regression Analysis </a> </li>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=25100&enhanced=Y" target = "_blank"> Data Structure and Algorithm </a> </li>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=STAT&term=CURRENT&cnbr=41700" target = "_blank"> Statistical Theory </a> </li>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=25000&enhanced=Y" target = "_blank"> Computer Architecture </a> </li>
            <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=TDM&term=CURRENT&cnbr=20100" target = "_blank"> Data Mine Seminar III </a> </li>
        </ul>
			</div>
		</div>
	</div>
	<!--Summer 2023-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Summer 2023
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__description">
				<ul>
         <li><a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=STAT&term=CURRENT&cnbr=41600" target = "_blank"> Probability </a> </li>
         <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=PSY&term=CURRENT&cnbr=12000" target = "_blank"> Elementary of Psychology</a></li>
        </ul>
			</div>
		</div>
	</div>
	<!--Spring 2023-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Spring 2023
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__description">
				<ul>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=17600&enhanced=Y" target = "_blank"> Data Engineering in Python</a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=24000&enhanced=Y" target = "_blank"> C Programming </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=18200&enhanced=Y" target = "_blank"> Discrete Mathematics </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=STAT&term=CURRENT&cnbr=35500" target = "_blank"> Statistics for Data Science </a> </li>
        <li><a href = "https://selfservice.mypurdue.purdue.edu/prod/bwckctlg.p_disp_course_detail?cat_term_in=201530&subj_code_in=MA&crse_numb_in=35100" target = "_blank"> Linear Algebra </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=38003&enhanced=Y" target = "_blank"> Python Programming </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=TDM&term=CURRENT&cnbr=10200" target = "_blank"> Data Mine Seminar II </a> </li>
        </ul>
			</div>
		</div>
	</div>
	<!--Fall 2022-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			Fall 2022
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__description">
				<ul>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=18000&enhanced=Y" target = "_black"> Object Oriented Programming (JAVA) </a></li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bwckctlg.p_disp_course_detail?cat_term_in=201510&subj_code_in=MA&crse_numb_in=26100" target = "_blank"> Multivariable Calculus </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=TDM&term=CURRENT&cnbr=11100"> 
            Corporate Partners I </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?subject=TDM&term=CURRENT&cnbr=10100" target = "_blank"> Data Mine Seminar I </a> </li>
        <li> <a href = "https://selfservice.mypurdue.purdue.edu/prod/bzwsrch.p_catalog_detail?term=202410&subject=CS&cnbr=19300&enhanced=Y" target = "_blank"> Tools </a> </li>
        </ul>
			</div>
		</div>
	</div>
</div>
</html>
