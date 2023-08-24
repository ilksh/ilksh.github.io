---
permalink: /project/
excerpt: "Project"
search: true
---
<html>
 <!-- <script src="https://kit.fontawesome.com/fc596df623.js" crossorigin="anonymous"></script> -->
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
  /* new style */ 
*,
*::after,
*::before {
  margin: 0;
  padding: 0;
  box-sizing: inherit;
  font-family: inherit; }
html {
  font-size: 62.5%;
  scroll-behavior: smooth; }
  @media only screen and (max-width: 75em) {
    html {
      font-size: 59%; } }
  @media only screen and (max-width: 56.25em) {
    html {
      font-size: 56%; } }
  @media only screen and (min-width: 112.5em) {
    html {
      font-size: 65%; } }
body {
  box-sizing: border-box;
  position: relative;
  line-height: 1.5;
  font-family: sans-serif;
  overflow-x: hidden;
  overflow-y: scroll;
  font-family: 'Source Sans Pro', sans-serif; }
a {
  text-decoration: none;
  color: inherit; }
li {
  list-style: none; }
input:focus,
button:focus,
a:focus,
textarea:focus {
  outline: none; }
button {
  border: none;
  cursor: pointer; }
textarea {
  resize: none; }
.heading-primary {
  font-size: 6rem;
  text-transform: uppercase;
  letter-spacing: 3px;
  text-align: center; }
  @media only screen and (max-width: 37.5em) {
    .heading-primary {
      font-size: 4.5rem; } }
.heading-sec__mb-bg {
  margin-bottom: 11rem; }
  @media only screen and (max-width: 56.25em) {
    .heading-sec__mb-bg {
      margin-bottom: 8rem; } }
.heading-sec__mb-med {
  margin-bottom: 9rem; }
  @media only screen and (max-width: 56.25em) {
    .heading-sec__mb-med {
      margin-bottom: 8rem; } }
.heading-sec__main {
  display: block;
  font-size: 4rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  letter-spacing: 3px;
  text-align: center;
  margin-bottom: 3.5rem;
  position: relative; }
  .heading-sec__main--lt {
    color: #fff; }
    .heading-sec__main--lt::after {
      content: '';
      background: #fff !important; }
  .heading-sec__main::after {
    content: '';
    position: absolute;
    top: calc(100% + 1.5rem);
    height: 5px;
    width: 3rem;
    background: #0062b9;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 5px; }
    @media only screen and (max-width: 37.5em) {
      .heading-sec__main::after {
        top: calc(100% + 1.2rem); } }
.heading-sec__sub {
  display: block;
  text-align: center;
  color: #777;
  font-size: 2rem;
  font-weight: 500;
  max-width: 80rem;
  margin: auto;
  line-height: 1.6; }
  @media only screen and (max-width: 37.5em) {
    .heading-sec__sub {
      font-size: 1.8rem; } }
  .heading-sec__sub--lt {
    color: #eee; }
.heading-sm {
  font-size: 2.2rem;
  text-transform: uppercase;
  letter-spacing: 1px; }
.main-container {
  max-width: 120rem;
  margin: auto;
  width: 92%; }
.btn {
  background: #fff;
  color: #333;
  text-transform: uppercase;
  letter-spacing: 2px;
  display: inline-block;
  font-weight: 700;
  border-radius: 5px;
  box-shadow: 0 5px 15px 0 rgba(0, 0, 0, 0.15);
  transition: transform .3s; }
  .btn:hover {
    transform: translateY(-3px); }
  .btn--bg {
    padding: 1.5rem 8rem;
    font-size: 2rem; }
  .btn--med {
    padding: 1.5rem 5rem;
    font-size: 1.6rem; }
  .btn--theme {
    background: #0062b9;
    color: #fff; }
  .btn--theme-inv {
    color: #0062b9;
    background: #fff;
    border: 2px solid #0062b9;
    box-shadow: none;
    padding: calc(1.5rem - 2px) calc(5rem - 2px); }
.sec-pad {
  padding: 12rem 0; }
  @media only screen and (max-width: 56.25em) {
    .sec-pad {
      padding: 8rem 0; } }
.text-primary {
  color: #fff;
  font-size: 2.2rem;
  text-align: center;
  width: 100%;
  line-height: 1.6; }
  @media only screen and (max-width: 37.5em) {
    .text-primary {
      font-size: 2rem; } }
.d-none {
  display: none; }
.home-hero {
  color: #fff;
  background: linear-gradient(to right, rgba(0, 98, 185, 0.8), rgba(0, 98, 185, 0.8)), url(../../assets/svg/common-bg.svg);
  background-position: center;
  height: 100vh;
  min-height: 80rem;
  max-height: 120rem;
  position: relative; }
  @media only screen and (max-width: 37.5em) {
    .home-hero {
      height: unset;
      min-height: unset; } }
  .home-hero__socials {
    position: absolute;
    top: 50%;
    border: 2px solid #eee;
    border-left: 2px solid #eee;
    transform: translateY(-50%); }
    @media only screen and (max-width: 56.25em) {
      .home-hero__socials {
        display: none; } }
  .home-hero__mouse-scroll-cont {
    position: absolute;
    bottom: 3%;
    left: 50%;
    transform: translateX(-50%); }
    @media only screen and (max-width: 37.5em) {
      .home-hero__mouse-scroll-cont {
        display: none; } }
  .home-hero__social {
    width: 5rem; }
  .home-hero__social-icon-link {
    width: 100%;
    display: block;
    padding: 1.2rem;
    border-bottom: 2px solid #eee;
    transition: background .3s; }
    .home-hero__social-icon-link:hover {
      background: rgba(255, 255, 255, 0.1); }
    .home-hero__social-icon-link--bd-none {
      border-bottom: 0; }
  .home-hero__social-icon {
    width: 100%; }
  .home-hero__content {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 90rem;
    width: 92%; }
    @media only screen and (max-width: 37.5em) {
      .home-hero__content {
        padding: 19rem 0 13rem 0;
        margin: auto;
        position: static;
        transform: translate(0, 0); } }
  .home-hero__info {
    margin: 3rem auto 0 auto;
    max-width: 80rem; }
  .home-hero__cta {
    margin-top: 5rem;
    text-align: center; }
.about {
  background: #fafafa; }
  .about__content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-gap: 10rem; }
    @media only screen and (max-width: 56.25em) {
      .about__content {
        grid-template-columns: 1fr;
        grid-gap: 8rem; } }
    .about__content-title {
      font-weight: 700;
      font-size: 2.8rem;
      margin-bottom: 3rem; }
      @media only screen and (max-width: 37.5em) {
        .about__content-title {
          font-size: 2.4rem; } }
    .about__content-details-para {
      font-size: 1.8rem;
      color: grey;
      max-width: 60rem;
      line-height: 1.7;
      margin-bottom: 1rem; }
      .about__content-details-para--hl {
        font-weight: 700;
        margin: 0 3px; }
      .about__content-details-para:last-child {
        margin-bottom: 4rem; }
.projects__row {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  grid-gap: 5rem;
  margin-bottom: 11rem; }
  @media only screen and (max-width: 56.25em) {
    .projects__row {
      grid-template-columns: 1fr;
      grid-gap: 2rem;
      margin-bottom: 8rem; } }
  @media only screen and (max-width: 56.25em) {
    .projects__row {
      text-align: center; } }
  .projects__row:last-child {
    margin-bottom: 0; }
  .projects__row-img-cont {
    overflow: hidden; }
  .projects__row-img {
    width: 100%;
    display: block;
    object-fit: cover; }
  .projects__row-content {
    padding: 2rem 0;
    display: flex;
    justify-content: center;
    flex-direction: column;
    align-items: flex-start; }
    @media only screen and (max-width: 56.25em) {
      .projects__row-content {
        align-items: center; } }
    .projects__row-content-title {
      font-weight: 700;
      font-size: 2.8rem;
      margin-bottom: 2rem; }
      @media only screen and (max-width: 37.5em) {
        .projects__row-content-title {
          font-size: 2.4rem; } }
    .projects__row-content-desc {
      font-size: 1.8rem;
      color: grey;
      max-width: 60rem;
      line-height: 1.7;
      margin-bottom: 3rem; }
      @media only screen and (max-width: 37.5em) {
        .projects__row-content-desc {
          font-size: 1.7rem; } }
.project-cs-hero {
  color: #fff;
  background: linear-gradient(to right, rgba(0, 98, 185, 0.8), rgba(0, 98, 185, 0.8)), url(../../assets/svg/common-bg.svg);
  background-size: cover;
  background-position: center;
  position: relative; }
  @media only screen and (max-width: 37.5em) {
    .project-cs-hero {
      height: unset;
      min-height: unset; } }
  .project-cs-hero__content {
    padding: 25rem 0 17rem 0;
    max-width: 90rem;
    width: 92%;
    margin: auto; }
    @media only screen and (max-width: 37.5em) {
      .project-cs-hero__content {
        padding: 19rem 0 13rem 0;
        margin: auto;
        position: static;
        transform: translate(0, 0); } }
  .project-cs-hero__info {
    margin: 3rem auto 0 auto;
    max-width: 80rem; }
  .project-cs-hero__cta {
    margin-top: 5rem;
    text-align: center; }
.project-details__content {
  padding: 8rem 0;
  max-width: 90rem;
  margin: auto; }
  .project-details__content-title {
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 3rem; }
    @media only screen and (max-width: 37.5em) {
      .project-details__content-title {
        font-size: 2.4rem; } }
.project-details__showcase-img-cont {
  width: 100%;
  margin-bottom: 6rem; }
.project-details__showcase-img {
  width: 100%; }
.project-details__content-main {
  width: 100%;
  max-width: 70rem;
  margin: auto; }
.project-details__desc {
  margin: 0 0 7rem 0; }
  .project-details__desc-para {
    font-size: 1.8rem;
    line-height: 1.7;
    color: grey;
    margin-bottom: 2rem; }
.project-details__tools-used {
  margin: 0 0 7rem 0; }
  .project-details__tools-used-list {
    display: flex;
    flex-wrap: wrap; }
  .project-details__tools-used-item {
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
    margin-right: 1.5rem;
    font-size: 1.6rem;
    background: rgba(153, 153, 153, 0.2);
    border-radius: 5px;
    font-weight: 600;
    color: #777; }
.project-details__links {
  margin: 0 0; }
  .project-details__links-btn {
    margin-right: 2rem; }
    @media only screen and (max-width: 37.5em) {
      .project-details__links-btn {
        margin-right: 0;
        width: 70%;
        margin-bottom: 2rem;
        text-align: center; } }
    .project-details__links-btn:last-child {
      margin: 0; }
      @media only screen and (max-width: 37.5em) {
        .project-details__links-btn:last-child {
          margin: 0; } }
.skills {
  display: flex;
  flex-wrap: wrap; }
  .skills__skill {
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
    margin-right: 1.5rem;
    font-size: 1.6rem;
    background: rgba(153, 153, 153, 0.2);
    border-radius: 5px;
    font-weight: 600;
    color: #777; }
.mouse {
  width: 25px;
  height: 40px;
  border: 2px solid #eee;
  border-radius: 60px;
  position: relative;
  overflow: hidden; }
  .mouse::before {
    content: '';
    width: 5px;
    height: 5px;
    position: absolute;
    top: 7px;
    left: 50%;
    transform: translateX(-50%);
    background-color: #eee;
    border-radius: 50%;
    opacity: 1;
    animation: wheel 1.3s infinite;
    -webkit-animation: wheel 1.3s infinite; }
@keyframes wheel {
  to {
    opacity: 0;
    top: 27px; } }
@-webkit-keyframes wheel {
  to {
    opacity: 0;
    top: 27px; } }
  /* Increase the maximum width of the main container */
.main-container {
  max-width: 400rem; /* Adjust this value as needed */
  margin: auto;
  width: 100%;
}
</style>
  <body>
    <section id="about" class="about sec-pad">
      <div class="main-container">
        <h2 class="heading heading-sec heading-sec__mb-med">
          <span class="heading-sec__main">About Me1</span>
          <span class="heading-sec__sub">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Hic facilis
            tempora explicabo quae quod deserunt eius sapiente
          </span>
        </h2>
        <div class="about__content">
          <div class="about__content-main">
            <h3 class="about__content-title">Get to know me!</h3>
            <div class="about__content-details">
              <p class="about__content-details-para">
                Hey! It's
                <strong>John Doe</strong>
                and I'm a <strong> Frontend Web Developer </strong> located in
                Los Angeles. I've done
                <strong> remote </strong>
                projects for agencies, consulted for startups, and collaborated
                with talented people to create
                <strong>digital products </strong>
                for both business and consumer use.
              </p>
              <p class="about__content-details-para">
                I'm a bit of a digital product junky. Over the years, I've used
                hundreds of web and mobile apps in different industries and
                verticals. Feel free to
                <strong>contact</strong> me here.
              </p>
            </div>
            <a href="./#contact" class="btn btn--med btn--theme dynamicBgClr"
              >Contact</a
            >
          </div>
          <div class="about__content-skills">
            <h3 class="about__content-title">My Skills</h3>
            <div class="skills">
              <div class="skills__skill">HTML</div>
              <div class="skills__skill">CSS</div>
              <div class="skills__skill">JavaScript</div>
              <div class="skills__skill">React</div>
              <div class="skills__skill">SASS</div>
              <div class="skills__skill">GIT</div>
              <div class="skills__skill">Shopify</div>
              <div class="skills__skill">Wordpress</div>
              <div class="skills__skill">Google ADS</div>
              <div class="skills__skill">Facebook Ads</div>
              <div class="skills__skill">Android</div>
              <div class="skills__skill">IOS</div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section id="projects" class="projects sec-pad">
      <div class="main-container">
        <h2 class="heading heading-sec heading-sec__mb-bg">
          <span class="heading-sec__main">Projects</span>
          <span class="heading-sec__sub">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Hic facilis
            tempora explicabo quae quod deserunt eius sapiente
          </span>
        </h2>
        <div class="projects__content">
          <div class="projects__row">
            <div class="projects__row-img-cont">
              <img
                src="/assets/image/MemoAppImg.png"
                alt="Software Screenshot"
                class="projects__row-img"
                loading="lazy"
              />
            </div>
            <div class="projects__row-content">
              <h3 class="projects__row-content-title">Project 1</h3>
              <p class="projects__row-content-desc">
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Hic
                facilis tempora, explicabo quae quod deserunt eius sapiente
                praesentium.
              </p>
              <a
                href="./project-1.html"
                class="btn btn--med btn--theme dynamicBgClr"
                target="_blank"
                >Case Study</a
              >
            </div>
          </div>
          <div class="projects__row">
            <div class="projects__row-img-cont">
              <img
                src="/assets/image/MemoAppImg.png"
                alt="Software Screenshot"
                class="projects__row-img"
                loading="lazy"
              />
            </div>
            <div class="projects__row-content">
              <h3 class="projects__row-content-title">Project 2</h3>
              <p class="projects__row-content-desc">
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Hic
                facilis tempora, explicabo quae quod deserunt eius sapiente
                praesentium.
              </p>
              <a
                href="./project-2.html"
                class="btn btn--med btn--theme dynamicBgClr"
                target="_blank"
                >Case Study</a
              >
            </div>
          </div>
          <div class="projects__row">
            <div class="projects__row-img-cont">
              <img
                src="/assets/image/MemoAppImg.png"
                alt="Software Screenshot"
                class="projects__row-img"
                loading="lazy"
              />
            </div>
            <div class="projects__row-content">
              <h3 class="projects__row-content-title">Project 3</h3>
              <p class="projects__row-content-desc">
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Hic
                facilis tempora, explicabo quae quod deserunt eius sapiente
                praesentium.
              </p>
              <a
                href="./project-3.html"
                class="btn btn--med btn--theme dynamicBgClr"
                target="_blank"
                >Case Study</a
              >
            </div>
          </div>
        </div>
      </div>
    </section>
  </body>
</html>
</html>