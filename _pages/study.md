---
title: "Aaron Kim"
permalink: /study/
excerpt: "About Me"
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
      color: #9251ac;
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
<h2>Super Mario Timeline</h2>
<h1>Initial launch dates of games in the Super Mario series.</h1>
<div class="timeline">
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			September 1985
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				Super Mario Brothers
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			June 1986
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Bros: The Lost Levels
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			October 1988
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Bros. 2
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			October 1988
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Bros. 3
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			April 1989
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				Super Mario Land
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 1990
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario World
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			October 1992
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Land: 6 Golden Coins
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			August 1995
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario World 2: Yoshi's Island
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			June 1996
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				Super Mario 64
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			July 2002
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Sunshine
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			May 2006
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				New Super Mario Bros
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 2007
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Galaxy
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 2009
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				New Super Mario Bros. Wii
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			May 2010
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Galaxy 2
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 2011
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario 3D Land
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			July 2012
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				New Super Mario Bros 2
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 2012
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				New Super Mario Bros. U
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			November 2013
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario 3D World
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			September 2015
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Maker
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			December 2016
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Run
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--first-->
	<div class="timeline__event  animated fadeInUp delay-3s timeline__event--type1">
		<div class="timeline__event__icon ">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			October 2017
		</div>
		<div class="timeline__event__content ">
			<div class="timeline__event__title">
				Super Mario Odyssey
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--second-->
	<div class="timeline__event animated fadeInUp delay-2s timeline__event--type2">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			June 2019
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Maker 2
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--third-->
	<div class="timeline__event animated fadeInUp delay-1s timeline__event--type3">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			February 2021
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario 3D World + Bowser's Fury
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>
	<!--forth-->
	<div class="timeline__event animated fadeInUp timeline__event--type1">
		<div class="timeline__event__icon">
			<!-- <i class="lni-sport"></i>-->
		</div>
		<div class="timeline__event__date">
			December 2016
		</div>
		<div class="timeline__event__content">
			<div class="timeline__event__title">
				Super Mario Run
			</div>
			<div class="timeline__event__description">
				<p>Lorem, ipsum dolor sit amet consectetur adipisicing elit. Vel, nam! Nam eveniet ut aliquam ab asperiores, accusamus iure veniam corporis incidunt reprehenderit accusantium id aut architecto harum quidem dolorem in!</p>
			</div>
		</div>
	</div>

</div>
</html>