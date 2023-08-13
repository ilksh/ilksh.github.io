---
title: "Aaron Kim"
permalink: /study/
excerpt: "About Me"
search: true
---

<html>
<style>
    $bg: #4D4545
    $color: #ED8D8D
    $font-stack: 'Lato', sans-serif
    html, body
    font: 100% $font-stack
    font-weight: 300
    height: 100%
    background-color: $bg
    .blue-bg
    background-color: $bg
    color: $color
    height: 100%
    .circle
    font-weight: bold
    padding: 15px 20px
    border-radius: 50%
    background-color: $color
    color: $bg
    max-height: 50px
    z-index: 2
    .how-it-works.row
    display: flex
    .col-2
        display: inline-flex
        align-self: stretch
        align-items: center
        justify-content: center
        &::after
            content: ''
            position: absolute
            border-left: 3px solid $color
            z-index: 1
    .col-2.bottom
        &::after
            height: 50%
            left: 50%
            top: 50%
    .col-2.full
        &::after
            height: 100%
            left: calc(50% - 3px)
    .col-2.top
        &::after
            height: 50%
            left: 50%
            top: 0
    .timeline
    div
        padding: 0
        height: 40px
    hr
        border-top: 3px solid $color
        margin: 0
        top: 17px
        position: relative
    .col-2
        display: flex
        overflow: hidden
    .corner
        border: 3px solid $color
        width: 100%
        position: relative
        border-radius: 15px
    .top-right
        left: 50%
        top: -50%
    .left-bottom
        left: -50%
        top: calc(50% - 3px)
    .top-left
        left: -50%
        top: -50%
    .right-bottom
        left: 50%
        top: calc(50% - 3px)
</style>
<div class="container-fluid blue-bg">
  <div class="container">
    <h2 class="pb-3 pt-2">Vertical Left-Right Timeline</h2>
    <!--first section-->
    <div class="row align-items-center how-it-works">
      <div class="col-2 text-center bottom">
        <div class="circle">1</div>
      </div>
      <div class="col-6">
        <h5>Fully Responsive</h5>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed porttitor gravida aliquam. Morbi orci urna, iaculis in ligula et, posuere interdum lectus.</p>
      </div>
    </div>
    <!--path between 1-2-->
    <div class="row timeline">
      <div class="col-2">
        <div class="corner top-right"></div>
      </div>
      <div class="col-8">
        <hr/>
      </div>
      <div class="col-2">
        <div class="corner left-bottom"></div>
      </div>
    </div>
    <!--second section-->
    <div class="row align-items-center justify-content-end how-it-works">
      <div class="col-6 text-right">
        <h5>Using Bootstrap</h5>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed porttitor gravida aliquam. Morbi orci urna, iaculis in ligula et, posuere interdum lectus.</p>
      </div>
      <div class="col-2 text-center full">
        <div class="circle">2</div>
      </div>
    </div>
    <!--path between 2-3-->
    <div class="row timeline">
      <div class="col-2">
        <div class="corner right-bottom"></div>
      </div>
      <div class="col-8">
        <hr/>
      </div>
      <div class="col-2">
        <div class="corner top-left"></div>
      </div>
    </div>
    <!--third section-->
    <div class="row align-items-center how-it-works">
      <div class="col-2 text-center top">
        <div class="circle">3</div>
      </div>
      <div class="col-6">
        <h5>Now with Pug and Sass</h5>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed porttitor gravida aliquam. Morbi orci urna, iaculis in ligula et, posuere interdum lectus.</p>
      </div>
    </div>
  </div>
</div>

</html>