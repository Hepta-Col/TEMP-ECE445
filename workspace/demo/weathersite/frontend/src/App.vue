<template>
  <div id="app" style="margin-top: 10px;">
    <!-- <img src="./assets/logo.png"> <br> -->
    <!-- <router-link to="/">Root</router-link>
    <router-link to="/data">Data</router-link>
    <router-link to="/weather">Weather</router-link> -->
    <!-- <router-link to="/hello">Hello</router-link>
    <hr> -->
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
body {
  max-width: 1000px;
  margin: 0 auto;
}
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
