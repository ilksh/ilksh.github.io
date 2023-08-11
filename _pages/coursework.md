---
title: "Course Work"
permalink: /course_work/
search: true
---

### new 5

<div class="container">
  <div class="timeline">
    <ul>
      <li>
        <div class="timeline-content">
          <h3 class="date">20th may, 2010</h3>
          <h1>Heading 1</h1>
          <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Consectetur tempora ab laudantium voluptatibus aut eos placeat laborum, quibusdam exercitationem labore.</p>
        </div>
      </li>
      <li>
        <div class="timeline-content">
          <h3 class="date">20th may, 2010</h3>
          <h1>Heading 2</h1>
          <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Consectetur tempora ab laudantium voluptatibus aut eos placeat laborum, quibusdam exercitationem labore.</p>
        </div>
      </li>
      <li>
        <div class="timeline-content">
          <h3 class="date">20th may, 2010</h3>
          <h1>Heading 3</h1>
          <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Consectetur tempora ab laudantium voluptatibus aut eos placeat laborum, quibusdam exercitationem labore.</p>
        </div>
      </li>
      <li>
        <div class="timeline-content">
          <h3 class="date">20th may, 2010</h3>
          <h1>Heading 4</h1>
          <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Consectetur tempora ab laudantium voluptatibus aut eos placeat laborum, quibusdam exercitationem labore.</p>
        </div>
      </li>
    </ul>
  </div>
</div>

<style>
/* Paste your CSS code here */
    @import url("https://fonts.googleapis.com/css2?family=Montserrat:wght@300;500&display=swap");
    *{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    html {
        font-family: "Montserrat";
    }
    .container {
        min-height: 100vh;
        width: 120%;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 100px 0;
        background-color: #111;
    }
    .timeline {
        width: 100%;
        height: auto;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
    }

    .timeline ul {
    list-style: none;
    }
    .timeline ul li {
    padding: 20px;
    background-color: #1e1f22;
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
    }
    .timeline ul li:last-child {
    margin-bottom: 0;
    }
    .timeline-content h1 {
    font-weight: 500;
    font-size: 25px;
    line-height: 30px;
    margin-bottom: 10px;
    }
    .timeline-content p {
    font-size: 16px;
    line-height: 30px;
    font-weight: 300;
    }
    .timeline-content .date {
    font-size: 12px;
    font-weight: 300;
    margin-bottom: 10px;
    letter-spacing: 2px;
    }
    @media only screen and (min-width: 768px) {
    .timeline:before {
        content: "";
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 2px;
        height: 100%;
        background-color: gray;
    }
    .timeline ul li {
        width: 50%;
        position: relative;
        margin-bottom: 50px;
    }
    .timeline ul li:nth-child(odd) {
        float: left;
        clear: right;
        transform: translateX(-30px);
        border-radius: 20px 0px 20px 20px;
    }
    .timeline ul li:nth-child(even) {
        float: right;
        clear: left;
        transform: translateX(30px);
        border-radius: 0px 20px 20px 20px;
    }
    .timeline ul li::before {
        content: "";
        position: absolute;
        height: 20px;
        width: 20px;
        border-radius: 50%;
        background-color: gray;
        top: 0px;
    }
    .timeline ul li:nth-child(odd)::before {
        transform: translate(50%, -50%);
        right: -30px;
    }
    .timeline ul li:nth-child(even)::before {
        transform: translate(-50%, -50%);
        left: -30px;
    }
    .timeline-content .date {
        position: absolute;
        top: -30px;
    }
    .timeline ul li:hover::before {
        background-color: aqua;
    }
    }

</style>