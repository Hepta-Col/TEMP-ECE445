<template>
  <div id="Weather">
   <!-- {{ msg_info }} -->

   <el-container>
  <el-header height = "40px">Welcome to our weather station</el-header>
  <el-container>
    <el-main>
      <div class = "content-box">
        <br>
        <h2 style="font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif, 'Arial Narrow', Arial, sans-serif ;margin-bottom: 0em; font-weight: bold;"> {{ "Haining Campus" }} <el-button @click = "refresh_now_data" icon="el-icon-refresh" circle size="mini"></el-button></h2>
        <h1 style="color:Tomato; font-size:xxx-large; margin-top: 0.2em; margin-bottom: 0.3em; font-weight: bold;"> {{ weather_data.temp }} {{ "\u2103" }} <a v-if="weather_data.rain"> <i class="icon iconfont icon-tianqi-yutian" style="font-size: 40px;"></i> </a> </h1>
        <h6 style="margin-top: 0em;"> {{ "Last update on" }}  {{ weather_data.day }} {{ "at" }} {{ weather_data.time }}</h6>
      </div>

    </el-main>
    <el-aside width=50%>
      <el-row></el-row>
      <el-row></el-row>
      <el-row>
          <el-col :span="9" :offset="2">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-shidu"></i> {{ " Humidity" }} <br/> </div>
                <div class = "br2" style="font-size:x-large; font-weight: 500;"> <br/> {{ weather_data.humi }} {{ "%" }} <br/> </div>
              </el-main>
            </el-container>
            <div class = "br3"><br/></div>
            <el-container>
              <el-tooltip class="item" effect="dark" :content="'Level ' + weather_data.wind_level + ': ' +weather_data.wind_msg" placement="bottom">
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-dafeng"></i> {{ " Wind" }} <br/> </div>
                <div class = "br2" style="font-size:x-large; font-weight: 500;"> <br/> <a v-if="[6,7,8,9,10,11].includes(weather_data.wind_level)"><i class="icon iconfont icon-alarm" style="font-size: 25px; color: brown;"></i></a> {{ weather_data.wind }} {{ "m/s" }} <br/> </div>
      
              </el-main>
              </el-tooltip>
            </el-container>
          </el-col>
          <el-col :span="10" :offset="2">
            <el-container>
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-daqiyali"></i> {{ " Pressure" }} <br/> </div>
                <div class = "br2"><br/></div>
                <div id="myChart" :style="{width: '180px', height: '126px'}"></div>
                <!-- <div class = "br3"><br/></div> -->
              </el-main>
              
            </el-container>

          </el-col>
      </el-row>
    </el-aside>
  </el-container>
  <el-footer  height = "200px">
    <el-row>
          <el-col :span="9" :offset="0">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br4" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-ziwaixian"></i> {{ " UV Index" }} <br/> </div>
                <div class = "br1" style="font-size:x-large; font-weight: 600;"> {{ weather_data.uv }} <span v-if="[11].includes(weather_data.uv)" style="font-weight: bold; margin-left: -8px;">+</span>
                  <br/> <span v-if="[0,1,2].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;"> Low </span> </span> 
                  <span v-if="[3,4,5].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Moderate</span></span>
                  <span v-if="[6,7].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">High</span></span> 
                  <span v-if="[8,9,10].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Very High</span></span>
                  <span v-if="[11].includes(weather_data.uv)"><span style="font-size: large; color: black; font-weight: 700;">Extreme</span></span>
                  <br/>
                </div>
                <div class = "br4" style="font-weight:200;"> 
                  <span v-if="[0,1,2].includes(weather_data.uv)"><span><i class="icon iconfont icon-mojing-" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Safely enjoy being outside. </span></span> 
                  <span v-if="[3,4,5].includes(weather_data.uv)"><span><i class="icon iconfont icon-huabanfuben" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Apply broad spectrum sunscreen. </span></span>
                  <span v-if="[6,7].includes(weather_data.uv)"><span><i class="icon iconfont icon-yusan" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Reduce time in the sun.</span></span> 
                  <span v-if="[8,9,10].includes(weather_data.uv)"><span><i class="icon iconfont icon-fangshaiyi" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Take extra precautions for skin.</span></span>
                  <span v-if="[11].includes(weather_data.uv)"><span><i class="icon iconfont icon-shineimdpi" style="font-size: 25px; margin-top: 10px;"></i></span> <span style="font-size: small; color: black; font-weight:bold;"> &nbsp; Avoid being outside during midday.</span></span>
                  <br/>
                </div>
                <!-- <el-progress type="line" :percentage="uv_percentage" :show-text="false"></el-progress>  -->
                <div class="cp-progress-main">
                  <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                    <div class="cp-progress-bar" :style="{ width: getPercentage+'%' ,background:getGradient,height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div>
                  </div>
                </div>
              </el-main>
            </el-container>
            <div class = "br3"><br/></div>
          </el-col>
          <el-col :span="14" :offset="1">
              <el-container>
                  <el-main style="background-color: rgb(253, 208, 215); border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                    <el-row>
                        <el-col :span="14">
                            <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-kongqizhiliang"></i> {{ " Air Quality" }} <br/> </div>
                            <div class = "br2"><br/></div>
                            <el-row :gutter="0">
                              <el-col :span="12" :marginBottom=-20>
                                <div class = "br3" style="font-weight: 500; margin-top: -10px; margin-bottom: -33px;"> <span style="font-size: medium; font-weight: 600; color: #46604e; font-style: italic;"> &nbsp; &nbsp; {{ "PM2.5" }} &nbsp; </span> </div>
                                <div id="pm25Chart" :style="{width: '130px', height: '142px', marginTop: '-20px'} "></div>
                                <div style="margin-bottom:-30px; margin-top: -15px;" class="br3"> <span style="font-size: x-small; font-weight: Bold;">&nbsp; &nbsp; &nbsp; {{ "Good" }} &nbsp; &nbsp; {{ "Hazardous" }}</span> <br/> <span style="margin-top: 5px;  margin-bottom:-30px ; font-size: medium; font-weight: Bold"> &emsp; {{ weather_data.pm25 }} {{ "ug/m³" }} </span> </div>
                              </el-col>
                              <el-col :span="11" :offset="1">
                                <div class = "br3" style="font-weight: 500; margin-top: -10px; margin-bottom: -33px;"> <span style="font-size: medium; font-weight: 600; color: #46604e; font-style: italic;"> &nbsp;  &nbsp;  {{ "PM10" }} &nbsp; </span>  </div>
                                <div id="pm100Chart" :style="{width:'130px', height: '142px', marginTop: '-20px'} "></div>
                                <div style="margin-bottom:-17px; margin-top: -15px;" class="br3"> <span style="font-size: x-small; font-weight: Bold;"> &nbsp; &ensp; {{ "Good" }} &nbsp; &nbsp; {{ "Hazardous" }}</span> <br/> <span style="margin-top: 5px;  margin-bottom:-30px ; font-size: medium; font-weight: Bold"> &emsp; {{ weather_data.pm100 }} {{ "ug/m³" }} </span> </div>
                              </el-col>
                            </el-row>
                        </el-col>
                        <el-col :span="8" :offset="2">
                                <!-- <div class = "br3" style="font-weight: 500; margin-top: -10px; margin-bottom: -33px;"> <span style="font-size: medium; font-weight: 600; color: #46604e; font-style: italic;">{{ "AQI" }} &nbsp; </span>  </div> -->
                                <div id="aqiChart" :style="{width:'170px', height: '180px', marginTop: '-20px'} "></div>
                                <div class = "br1" style="font-weight:200; margin-top: -25px; margin-left: -20px; margin-right: -15px; margin-bottom: -20px; line-height: normal; font-size: smaller;"> 
                                  <div class = "br3" v-if="[1].includes(weather_data.aqi_level)" style="font-size: small; font-weight:bold; margin-bottom: -14px; color: #287937; font-weight: bold;"> Air pollution poses little risk. <br/> <span><i class="icon iconfont icon-huwai" style="font-size: 18px; margin-top: 10px;"></i></span> Enjor outdoor activities.</div> 
                                  <span v-if="[2].includes(weather_data.aqi_level)"><span style="font-size: small; color: rgb(144, 184, 64); font-weight:bold;"> <span><i class="icon iconfont icon-yinger" style="font-size: 23px; margin-top: 0px;"></i></span> Risk for extremely sensitive children and adults. </span></span>
                                  <span v-if="[3].includes(weather_data.aqi_level)"><span style="font-size: small; color: orange; font-weight:bold;">  <span><i class="icon iconfont icon-ertongerbihou-tunyankunnan" style="font-size: 25px; margin-top: 0px;"></i></span> Sensitive groups may experience health affects.</span></span> 
                                  <span class = "br3" v-if="[4].includes(weather_data.aqi_level)"><span style="font-size: small; font-weight:bold; color:white"> <span><i class="icon iconfont icon-kouzhao1" style="font-size: 20px; margin-top: 5px;"></i></span> <span style="color: red; font-weight: bold;">The general public may experience health effects. </span></span></span>
                                  <div class = "br3" style="margin-bottom:-15px;" v-if="[5].includes(weather_data.aqi_level)" ><span style="font-size: small; color: red; font-weight:bold;"> <span><i class="icon iconfont icon-jingshi" style="font-size: 23px; margin-top: 0px;"></i></span> <span style="color: purple; font-weight: bold;">Health alert!</span> <br/> <span style="color: purple; font-weight: bold;">Risk increases for everyone.</span></span></div>
                                  <span v-if="[6].includes(weather_data.aqi_level)"><span style="font-size: small; color: brown; font-weight:bold;"> <span><i class="icon iconfont icon-EARLY_WARNING" style="font-size: 23px; margin-top: 2px;"></i></span> Health WARNING. Everyone should avoid outdoor exertion.</span></span>
                                  <br/>
                                </div>
                        </el-col>
                    </el-row>
                    <!-- <div class = "br3"><br/></div> -->
                  </el-main>
                </el-container>

          </el-col>
      </el-row>
  </el-footer>
</el-container>
  <div class="br2"><br/></div>
 <hr/>
  <div class="br2" style="margin-bottom: -5px;"><br/></div>
  <div class = "br5" style="margin-left: 20px; margin-right: -20px; margin-bottom: -60px;">
    <el-row>
          <el-col :span="15" :offset="0">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              <el-main style="background-color: #C8E7F5; border-radius: 10px; margin-top: 0px; margin-bottom: -40px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <el-row>
                  <el-col :span = "14">
                    <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-jiageyuce"></i> {{ " Hourly Prediction" }} <el-button @click = "refresh_hourly_data" icon="el-icon-refresh" circle size="mini"></el-button> <br/> </div>
                    <!-- <div class = "br2"><br/></div> -->
                    <el-row>
                      <el-col :span = "12">
                        <div class = "br3" style="font-weight: 500; margin-top: 0px; margin-bottom: -10px; "> <span style="font-size: medium; font-weight: 600; font-style: italic;"> <i class="icon iconfont icon-shidu"></i> {{ "Humidity" }} &nbsp; </span> </div>
                        <div id="hourlyHumiChart" :style="{width: '240px', height: '240px', marginTop: '-20px', marginLeft: '-35px', marginBottom: '-40px',}  "></div>
                      </el-col>
                      <el-col :span = "11" :offset="1">
                        <div class = "br3" style="font-weight: 500; margin-top: 0px; margin-bottom: -10px; "> <span style="font-size: medium; font-weight: 600; font-style: italic;"> &emsp; &emsp; <i class="icon iconfont icon-daqiyali"></i>{{ "Pressure" }} &nbsp; </span> </div>
                        <div id="hourlyPresChart" :style="{width: '240px', height: '240px', marginTop: '-20px', marginLeft: '-20px', marginBottom: '-40px',}  "></div>
                      </el-col>
                    </el-row>
                  </el-col>
                  <el-col :span = "8" :offset = "2" class="br2">
                    <div class = "br3" style="font-weight: 500; margin-top: -10px; margin-bottom: -10px; "> <span style="font-size: medium; font-weight: 600; font-style: italic;"> <i class="icon iconfont icon-wendu1"></i> {{ "Temperature" }} &nbsp; </span> </div>
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left;"> &emsp; {{ hourly_pred.Hour1 }} &nbsp; <span style="color:rgb(66, 173, 203); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MinT1.length == 1"> &nbsp;</span> {{ hourly_pred.MinT1 }}  {{ "\u2103" }} </span> &nbsp; <i class="icon iconfont icon-fengefu" style="color: orange;"></i> &nbsp; <span style="color:rgb(218, 69, 58); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MaxT1.length == 1"> &nbsp;</span> {{ hourly_pred.MaxT1 }} {{ "\u2103" }}</span> </div>
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left;"> &emsp; {{ hourly_pred.Hour2 }} &nbsp; <span style="color:rgb(66, 173, 203); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MinT2.length == 1"> &nbsp;</span> {{ hourly_pred.MinT2 }}  {{ "\u2103" }} </span> &nbsp; <i class="icon iconfont icon-fengefu" style="color: orange;"></i> &nbsp; <span style="color:rgb(218, 69, 58); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MaxT2.length == 1"> &nbsp;</span> {{ hourly_pred.MaxT2 }} {{ "\u2103" }}</span> </div>
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left;"> &emsp; {{ hourly_pred.Hour3 }} &nbsp; <span style="color:rgb(66, 173, 203); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MinT3.length == 1"> &nbsp;</span> {{ hourly_pred.MinT3 }}  {{ "\u2103" }} </span> &nbsp; <i class="icon iconfont icon-fengefu" style="color: orange;"></i> &nbsp; <span style="color:rgb(218, 69, 58); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MaxT3.length == 1"> &nbsp;</span> {{ hourly_pred.MaxT3 }} {{ "\u2103" }}</span> </div>
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left;"> &emsp; {{ hourly_pred.Hour4 }} &nbsp; <span style="color:rgb(66, 173, 203); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MinT4.length == 1"> &nbsp;</span> {{ hourly_pred.MinT4 }}  {{ "\u2103" }} </span> &nbsp; <i class="icon iconfont icon-fengefu" style="color: orange;"></i> &nbsp; <span style="color:rgb(218, 69, 58); font-weight: Bold; font-size: medium;"> <span v-if="hourly_pred.MaxT4.length == 1"> &nbsp;</span> {{ hourly_pred.MaxT4 }} {{ "\u2103" }}</span> </div>
                    <br>
                    <hr style="background-color:aliceblue; height:1px; border: none;"/>
                    <div class = "br3" style="font-weight: 500; margin-top: 0px; margin-bottom: -10px; "> <span style="font-size: medium; font-weight: 600; font-style: italic;"> &nbsp; <i class="icon iconfont icon-dafeng"></i> {{ "Wind" }}</span> </div>
                    <div class="right">
                    <el-tooltip class="item" effect="dark" :content="'Level ' + hourly_pred.Wind_level1 + ': ' + hourly_pred.Wind_msg1" placement="right">
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left; margin-left: 20px; margin-right: 0px;">{{ hourly_pred.Hour1 }} &nbsp; <span style="color:dimgrey; font-weight: Bold; font-size: medium; margin-left: 30px; margin-right: -20px;">{{ hourly_pred.Wind1 }}  {{ "m/s" }} <span style="text-align: end; font-weight: bold;"> &nbsp; {{ "|" }} &nbsp; <i v-if="hourly_pred.des1==0" class="icon iconfont icon-w_tianqi" style="color: orange;"></i>   <i v-if="hourly_pred.des1==1" class="icon iconfont icon-zhongyu" style="color: blue;"></i> </span> </span> </div>
                    </el-tooltip> 
                    <el-tooltip class="item" effect="dark" :content="'Level ' + hourly_pred.Wind_level2 + ': ' + hourly_pred.Wind_msg2" placement="right">
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left; margin-left: 20px; margin-right: 0px;">{{ hourly_pred.Hour2 }} &nbsp; <span style="color:dimgrey; font-weight: Bold; font-size: medium; margin-left: 30px; margin-right: -20px;">{{ hourly_pred.Wind2 }}  {{ "m/s" }} <span style="text-align: right; font-weight: bold;"> &nbsp; {{ "|" }} &nbsp; <i v-if="hourly_pred.des2==0" class="icon iconfont icon-w_tianqi" style="color: orange;"></i>   <i v-if="hourly_pred.des2==1" class="icon iconfont icon-zhongyu" style="color: blue;"></i> </span> </span> </div>
                    </el-tooltip> 
                    <el-tooltip class="item" effect="dark" :content="'Level ' + hourly_pred.Wind_level3 + ': ' + hourly_pred.Wind_msg3" placement="right">
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left; margin-left: 20px; margin-right: 0px;">{{ hourly_pred.Hour3 }} &nbsp; <span style="color:dimgrey; font-weight: Bold; font-size: medium; margin-left: 30px; margin-right: -20px;">{{ hourly_pred.Wind3 }}  {{ "m/s" }} <span style="text-align: right; font-weight: bold;"> &nbsp; {{ "|" }} &nbsp; <i v-if="hourly_pred.des3==0" class="icon iconfont icon-w_tianqi" style="color: orange;"></i>   <i v-if="hourly_pred.des3==1" class="icon iconfont icon-zhongyu" style="color: blue;"></i> </span> </span> </div>
                    </el-tooltip> 
                    <el-tooltip class="item" effect="dark" :content="'Level ' + hourly_pred.Wind_level3 + ': ' + hourly_pred.Wind_msg3" placement="right">
                    <div class = "br3" style="margin-top: 10px; margin-bottom: -10px; font-size: medium; font-weight: Bold; text-align: left; margin-left: 20px; margin-right: 0px;">{{ hourly_pred.Hour4 }} &nbsp; <span style="color:dimgrey; font-weight: Bold; font-size: medium; margin-left: 30px; margin-right: -20px;">{{ hourly_pred.Wind4 }}  {{ "m/s" }} <span style="text-align: end; font-weight: bold;"> &nbsp; {{ "|" }} &nbsp; <i v-if="hourly_pred.des4==0" class="icon iconfont icon-w_tianqi" style="color: orange;"></i>   <i v-if="hourly_pred.des4==1" class="icon iconfont icon-zhongyu" style="color: blue;"></i> </span> </span> </div>
                    </el-tooltip> 
                    </div>
                  </el-col>
              </el-row>
              </el-main>
            </el-container>
          </el-col>

          <el-col :span="7" :offset="1">
            <el-container>
              <!-- <el-header height = "40px" style="background-color: red; text-align:left; ">Humidity</el-header> -->
              <el-main style="background-color: #c4e3ff; border-radius: 10px; margin-top: 0px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1)">
                <div class = "br1" style="text-align: left; margin-top: -10px; font-size: large; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif; font-weight:550;"><i class="icon iconfont icon-jiageyuce"></i> {{ " Daily Prediction" }} <el-button @click = "refresh_daily_data" icon="el-icon-refresh" circle size="mini"></el-button> <br/> </div>
                <div class = "br2"><br/></div>
                <!-- <div>{{ daily_pred }}</div> -->
                <!-- <div><span>{{ dayDelta1 }}</span> <span>{{ dayDelta2 }}</span> <span>{{ dayDelta3 }}</span></div> -->
                <el-row>
                    <el-col :span="11" :offset="0">
                      <div style="margin-left: -35px; margin-right: 5px; font-weight: bold; font-size: medium; text-align: right;">{{ daily_pred.DAY1[0] }} &thinsp; {{ daily_pred.DAY1[1] }} {{ "\u2103" }} </div>
                      <div style="margin-left: -35px; margin-right: 5px; font-weight: bold; font-size: medium; text-align: right;">{{ daily_pred.DAY2[0] }} &thinsp; {{ daily_pred.DAY2[1] }} {{ "\u2103" }} </div>
                      <div style="margin-left: -35px; margin-right: 5px; font-weight: bold; font-size: medium; text-align: right;">{{ daily_pred.DAY3[0] }} &thinsp; {{ daily_pred.DAY3[1] }} {{ "\u2103" }} </div>
                    </el-col>
                    <el-col :span="9" :offset="0">
                      <el-row>
                        <el-col :span = preLength1 :offset="0">
                          <div class="cp-progress-main br1" style="margin-left: 3px;">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <div class="cp-progress-bar" :style="{ width: '100%' , height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                              <!-- <div :style="{ width: dayPercentage1+'%' ,background:dayGradient1,height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                            </div>
                          </div>
                        </el-col>
                        <el-col :span = "24-preLength1" :offset="0">
                          <div class="cp-progress-main">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <!-- <div class="cp-progress-bar" :style="{ width: '10%' , height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                              <div :style="{ width: dayPercentage1+'%' ,background:dayGradient1,height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                            </div>
                          </div>
                        </el-col>
                      </el-row>
                      <el-row>
                        <el-col :span = preLength2 :offset="0">
                          <div class="cp-progress-main br1" style="margin-left: 3px;">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <div class="cp-progress-bar" :style="{ width: '100%' , height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                              <!-- <div :style="{ width: dayPercentage1+'%' ,background:dayGradient1,height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                            </div>
                          </div>
                        </el-col>
                        <el-col :span = "24-preLength2" :offset="0">
                          <div class="cp-progress-main">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <!-- <div class="cp-progress-bar" :style="{ width: '10%' , height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                              <div :style="{ width: dayPercentage2+'%' ,background:dayGradient2,height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                            </div>
                          </div>
                        </el-col>
                      </el-row>
                      <el-row>
                        <el-col :span = preLength3 :offset="0">
                          <div class="cp-progress-main br1" style="margin-left: 3px;">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <div class="cp-progress-bar" :style="{ width: '100%' , height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                              <!-- <div :style="{ width: dayPercentage1+'%' ,background:dayGradient1,height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                            </div>
                          </div>
                        </el-col>
                        <el-col :span = "24-preLength3" :offset="0">
                          <div class="cp-progress-main">
                            <div class="cp-progress-bg" :style="{ 'border-radius': bRadius+'px'}">
                              <!-- <div class="cp-progress-bar" :style="{ width: '10%' , height:strokeWidth+'px' ,'border-radius': bRadius+'px'}"></div> -->
                              <div :style="{ width: dayPercentage3+'%' ,background:dayGradient3,height:strokeWidth2+'px' ,'border-radius': bRadius+'px'}"></div>
                            </div>
                          </div>
                        </el-col>
                      </el-row>
                    </el-col>
                    <el-col :span="3" :offset="1">
                      <div style="margin-left: -28px; margin-right: -20px; font-weight: bold;">&nbsp; {{ daily_pred.DAY1[2] }} {{ "\u2103" }} </div>
                      <div style="margin-left: -28px; margin-right: -20px; font-weight: bold;">&nbsp; {{ daily_pred.DAY2[2] }} {{ "\u2103" }} </div>
                      <div style="margin-left: -28px; margin-right: -20px; font-weight: bold;">&nbsp; {{ daily_pred.DAY3[2] }} {{ "\u2103" }} </div>
                    </el-col>
                </el-row>
              </el-main>
              <el-footer>
                <br>
                <div class = "br2" style="margin-left:-20px;  margin-right:-20px; text-align: left; font-weight: bold; font-size:small; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif, 'Arial Narrow', Arial, sans-serif ;">* Last hourly prediction update on  {{ hourly_pred.day }} {{ "at" }} {{ hourly_pred.time }}</div>
                <div  class = "br2" style="margin-left:-20px; margin-right:-10px;  text-align: left; font-weight: bold; font-size:small; font-family:'Trebuchet MS', 'Lucida Sans Unicode', 'Lucida Grande', 'Lucida Sans', Arial, sans-serif, 'Arial Narrow', Arial, sans-serif ;">* Last daily prediction update on  {{ daily_pred.day }} {{ "at" }} {{ daily_pred.time }}</div>
              </el-footer>
            </el-container>
          </el-col>

    </el-row>
  <br>
  </div>
  

   <!-- <el-row :gutter="20">
   <el-col :span="16"><div class="grid-content bg-purple"> {{ "Temperature:" }} {{ weather_data.temp }} {{ "\u2103" }}</div></el-col>
   <el-col :span="8"><div class="grid-content bg-purple"> {{ "Humidity: " }} {{ weather_data.humi }} {{ "%" }}</div></el-col>
   </el-row>

   <el-row :gutter="20">
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Temperature:" }} {{ weather_data.temp }} {{ "\u2103" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Humidity: " }} {{ weather_data.humi }} {{ "%" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple">{{ "Pressure: " }} {{ weather_data.pres }} {{ "hPa" }}</div></el-col>
   <el-col :span="6"><div class="grid-content bg-purple"> {{ "Wind: " }} {{ weather_data.wind }} {{ "m/s" }}</div></el-col>
   </el-row> -->
  </div>
  

</template>

<script>
var echarts = require('echarts');
export default {
  name: 'Weather',
  data () {
    return {
      msg_info: 'Play with elementUI',
      weather_data: "",
      hourly_pred: "",
      daily_pred: "",
      draw_pressure: "",
      draw_pm25: "",
      draw_pm100: "",
      draw_aqi_val: "",
      uv_percentage: "",
      dayDelta1: "",
      dayDelta2: "",
      dayDelta3: "",
      preLength1: "",
      preLength2: "",
      preLength3: "",
      // pressure_percentage: "",
      colors: [
          {color: '#f56c6c', percentage: 20},
          {color: '#e6a23c', percentage: 40},
          {color: '#5cb87a', percentage: 60},
          {color: '#1989fa', percentage: 80},
          {color: '#6f7ad3', percentage: 100}
        ],
    }
  },
  computed: {
    // 通过比例 获取 百分比
    getPercentage () {
      if (this.uv_percentage < 0) {
        return 0
      } else if (this.uv_percentage > 1) {
        return 100
      } else {
        // console.log(this.percentage)
        return parseInt((this.uv_percentage + 0.000006) * 100)
      }
    },
    // 获取 进度条颜色对象
    getGradient () {
      let linecolor = this.getColorItem(this.uv_percentage)
      if (linecolor) {
        return 'linear-gradient(90deg,' + linecolor.s + ',' + linecolor.e + ')'
      } else {
        return ''
      }
    },
    dayPercentage1 () {
      // console.log(this.daily_pred["max"])
      // console.log(parseInt((((this.dayDelta1)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100))
      return parseInt((((this.dayDelta1)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
    },
    dayPercentage2 () {
      return parseInt((((this.dayDelta2)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
    },
    dayPercentage3 () {
      return parseInt((((this.dayDelta3)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
    },
    prePercentage1 (){
      console.log(this.daily_pred["DAY1"][1])
      return parseInt((((this.daily_pred["DAY1"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min)) + 0.000006) * 100)
    },
    // prePercentage2 (){
    //   return parseInt((((this.daily_pred.DAY2[1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min)) + 0.000006) * 100)
    // },
    // prePercentage3 (){
    //   return parseInt((((this.daily_pred.DAY3[1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min)) + 0.000006) * 100)
    // },
    // delta: percentage of the colored part
    dayGradient1 () {
      let delta = parseInt((((this.dayDelta1)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
      console.log(delta)
      let linecolor = this.getColorItem2(delta)
      if (linecolor) {
        return 'linear-gradient(90deg,' + linecolor.s + ',' + linecolor.e + ')'
      } else {
        return ''
      }
    },
    dayGradient2 () {
      let delta = parseInt((((this.dayDelta2)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
      console.log(delta)
      let linecolor = this.getColorItem2(delta)
      if (linecolor) {
        return 'linear-gradient(90deg,' + linecolor.s + ',' + linecolor.e + ')'
      } else {
        return ''
      }
    },
    dayGradient3 () {
      let delta = parseInt((((this.dayDelta3)/(this.daily_pred["max"] - this.daily_pred["min"])) + 0.000006) * 100)
      console.log(delta)
      let linecolor = this.getColorItem2(delta)
      if (linecolor) {
        return 'linear-gradient(90deg,' + linecolor.s + ',' + linecolor.e + ')'
      } else {
        return ''
      }
    },

  },
  mounted(){
    setTimeout(()=>{
                console.log(this.drawline());
                console.log(this.draw_pm1());
                console.log(this.draw_pm2());
                console.log(this.draw_aqi()); // undefined
                console.log(this.draw_hour_humi());
                console.log(this.draw_hour_pres());
    },50)
  },
  created() {
    this.$axios.get("http://localhost:8000/data/")
      .then(response =>{
        this.weather_data = response.data
        this.draw_pressure = this.weather_data.pres
        this.draw_pm25 = this.weather_data.iaqi1
        this.draw_pm100 = this.weather_data.iaqi2
        this.draw_aqi_val = this.weather_data.aqi
        this.uv_percentage = (this.weather_data.uv +1) /12
        // this.drawline()
        // this.draw_pm1()
        // this.draw_pm2()
        // this.draw_aqi()
        // this.draw_hour_humi()
        // this.draw_hour_pres()
      })
      .catch((error) => {
        console.log("error in realtime data retrieval")
        console.log(error);
      });
      this.$axios.get("http://localhost:8000/data/hourlypredict/")
      .then(res =>{
        this.hourly_pred = res.data
      })
      .catch((error) => {
        console.log("error in hourly prediction retrieval")
        console.log(error);
      });
      this.$axios.get("http://localhost:8000/data/dailypredict/")
      .then(res =>{
        this.daily_pred = res.data
        // console.log(this.daily_pred)
        this.dayDelta1 = this.daily_pred["DAY1"][2]- this.daily_pred["DAY1"][1]
        this.dayDelta2 = this.daily_pred["DAY2"][2]- this.daily_pred["DAY2"][1]
        this.dayDelta3 = this.daily_pred["DAY3"][2]- this.daily_pred["DAY3"][1]
        this.preLength1 = parseInt(((this.daily_pred["DAY1"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
        this.preLength2 = parseInt(((this.daily_pred["DAY2"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
        this.preLength3 = parseInt(((this.daily_pred["DAY3"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
      })
      .catch((error) => {
        console.log("error in daily prediction retrieval")
        console.log(error);
      });
  },
  methods: {
    refresh_now_data(event){
      console.log("Refresh real-time data");
      this.$axios.get("http://localhost:8000/data/")
      .then(response =>{
        this.weather_data = response.data
        this.draw_pressure = this.weather_data.pres
        this.draw_pm25 = this.weather_data.iaqi1
        this.draw_pm100 = this.weather_data.iaqi2
        this.draw_aqi_val = this.weather_data.aqi
        this.uv_percentage = (this.weather_data.uv +1) /12
        // this.drawline()
        // this.draw_pm1()
        // this.draw_pm2()
        // this.draw_aqi()
        // this.draw_hour_humi()
        // this.draw_hour_pres()
      })
      .catch((error) => {
        console.log("error in realtime data retrieval")
        console.log(error);
      });
    },
    refresh_hourly_data(event){
      console.log("Refresh hourly prediction data");
      this.$axios.get("http://localhost:8000/data/hourlypredict/")
      .then(res =>{
        this.hourly_pred = res.data
      })
      .catch((error) => {
        console.log("error in hourly prediction retrieval")
        console.log(error);
      });
    },
    refresh_daily_data(event){
      console.log("Refresh daily prediction data")
      this.$axios.get("http://localhost:8000/data/dailypredict/")
      .then(res =>{
        this.daily_pred = res.data
        // console.log(this.daily_pred)
        this.dayDelta1 = this.daily_pred["DAY1"][2]- this.daily_pred["DAY1"][1]
        this.dayDelta2 = this.daily_pred["DAY2"][2]- this.daily_pred["DAY2"][1]
        this.dayDelta3 = this.daily_pred["DAY3"][2]- this.daily_pred["DAY3"][1]
        this.preLength1 = parseInt(((this.daily_pred["DAY1"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
        this.preLength2 = parseInt(((this.daily_pred["DAY2"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
        this.preLength3 = parseInt(((this.daily_pred["DAY3"][1]-this.daily_pred.min)/(this.daily_pred.max - this.daily_pred.min))*24)
      })
      .catch((error) => {
        console.log("error in daily prediction retrieval")
        console.log(error);
      });
    },
    drawline(){
      let myChart = this.$echarts.init(document.getElementById('myChart'))
      myChart.setOption({
        // tooltip: {
        // formatter: '{a} <br/>{b} : {c}%'
        // },
      series: [{
            name: 'pressure',
            type: 'gauge',
            max: 1060,
            min: 860,
            startAngle: 225,
            endAngle: -45,
            progress: {
              show: false,
            },
            pointer: { // 指针样式
                width: 3,
                length: '60%',
                shadowBlur: 5,
                show: true,
                // offsetCenter: ['-20%', 0],
                itemStyle: {
                    color: '#99a9bf',
                    padding: [50,0,0,0]
                }
            },
            detail: {
                valueAnimation: true,
                formatter: ['{value}', '{a|hPa}'].join('\n'),
                fontSize: 20,
                width: '100%',
                height: '30%',
                padding: [-80, 0, 0, 0],
                rich: {
                    a: {
                        color: '#68A54A',
                        fontSize: 16,
                        padding: [3, 0, 10, 0],
                    }
                }
            },
            data: [{
                value: this.draw_pressure,
            }],
            axisLine: {
                lineStyle: {
                    color: [ //数组第一个属性是颜色所占line百分比
                        [0.3, "#49afff"],
                        [0.7, "#68A54A"],
                        [1, "#f56c6c"],

                    ],
                    width: 10
                }
            },
            splitLine: {
                // length: 5,
                // lineStyle: {
                //     width: 3
                // }
                show: false
            },
            axisLabel: {
                distance: -40,
                show: true,
                formatter: function(value) {
                    if (value === 860) {
                        return "Low"
                    }
                    if(value == 1060){
                        return "High"
                    }
                    return ''
                },
                //   padding: '8 0 0 0'
                lineHeight: -60,
                fontSize: 15,
                fontWeight: "bold"
            },
            axisTick: {
                show: false
            },
            animation: true,
            animationEasingUpdate: "quadraticIn"
        },
      ],
      grid: {
                top: "0%",
                bottom: "0%"
              },
      });
    },
    draw_pm1 () {
      let pm25Chart = this.$echarts.init(document.getElementById('pm25Chart'))
      pm25Chart.setOption({
        tooltip: {
        formatter: "{a} {b} : {c}"
    },
    // toolbox: {
    //     feature: {
    //         restore: {},
    //         saveAsImage: {}
    //     }
    // },
    series: [{
        name: 'PM2.5',
        type: 'gauge',
        max: 500,
        center: [80,88],
        progress:{
            show: false
        },
        pointer: {
            show: true,
            width: 3,
            itemStyle: {
                color: '#8ec3e0',
            }
        },
        detail: {
            valueAnimation: true,
            formatter: ['{a|IAQI}', '{value}'].join('\n'),
            fontSize: 16,
            padding: [-42,0,0,0],
            fontWeight: "bold",
            rich: {
                    a: {
                        color: '#68A54A',
                        fontSize: 18,
                        padding: [13, 0, 10, 0],
                        fontWeight: "bold"
                    }
                }
        },
        data: [{
            value: this.draw_pm25,
            // name: 'IAQI'
        }],
        axisLine: {
            show: true,
            width: 5,
            lineStyle: {
                color: [
                    [0.1,"#8fce00"],
                     [0.2,"#ffd629"],
                     [0.3,"#f6b26b"],
                     [0.4,"#cc0000"],
                     [0.6,"#a64d79"],
                     [1,"#660000"]
                ]

            }
        },
        axisLabel: {
                distance: -6,
                show: true,
                formatter: function(value) {
                    // if (value === 0) {
                    //     return "Good"
                    // }
                    // if(value == 500){
                    //     return "Hazardous"
                    // }
                    // if(value == 100 || value == 200 || value == 300 || value == 400){
                    //   return value
                    // }
                },
                //   padding: '8 0 0 0'
                lineHeight: 12,
                fontSize:12,
                fontWeight: "bolder"
            },
            axisTick: {
            show: true,
            lineStyle: {
                color: 'auto',
                width: 0
            },
            length: -10
        },
        splitLine: {
            show: false,
            length: -10,
            distance: 0,
            lineStyle: {
                color: '#fff6db',
                width: 1
        },
        },
    }]

      });
    },
    draw_pm2 () {
      let pm100Chart = this.$echarts.init(document.getElementById('pm100Chart'))
      pm100Chart.setOption({
        tooltip: {
        formatter: "{a} {b} : {c}"
    },
    // toolbox: {
    //     feature: {
    //         restore: {},
    //         saveAsImage: {}
    //     }
    // },
    series: [{
        name: 'PM10',
        type: 'gauge',
        max: 500,
        center: [75,88],
        progress:{
            show: false
        },
        pointer: {
            show: true,
            width: 3,
            itemStyle: {
                color: '#8ec3e0',
            }
        },
        detail: {
            valueAnimation: true,
            formatter: ['{a|IAQI}', '{value}'].join('\n'),
            fontSize: 16,
            padding: [-42,0,0,0],
            fontWeight: "bold",
            rich: {
                    a: {
                        color: '#68A54A',
                        fontSize: 18,
                        padding: [13, 0, 10, 0],
                        fontWeight: "bold"
                    }
                }
        },
        data: [{
            value: this.draw_pm100,
            // name: 'IAQI'
        }],
        axisLine: {
            show: true,
            width: 5,
            lineStyle: {
                color: [
                [0.1,"#8fce00"],
                     [0.2,"#ffd629"],
                     [0.3,"#f6b26b"],
                     [0.4,"#cc0000"],
                     [0.6,"#a64d79"],
                     [1,"#660000"]
                ]

            }
        },
        axisLabel: {
                distance: -6,
                show: true,
                formatter: function(value) {
                    // if (value === 0) {
                    //     return "Good"
                    // }
                    // if(value == 500){
                    //     return "Hazardous"
                    // }
                    // if(value == 100 || value == 200 || value == 300 || value == 400){
                    //   return value
                    // }
                },
                //   padding: '8 0 0 0'
                lineHeight: 12,
                fontSize:12,
                fontWeight: "bolder"
            },
            axisTick: {
            show: true,
            lineStyle: {
                color: 'auto',
                width: 0
            },
            length: -10
        },
        splitLine: {
            show: false,
            length: -10,
            distance: 0,
            lineStyle: {
                color: '#fff6db',
                width: 1
        },
        },
    }]

      });
    },
    draw_aqi () {
      let aqiChart = this.$echarts.init(document.getElementById('aqiChart'))
      aqiChart.setOption({
        tooltip: {
        formatter: "{a} {b} : {c}"
    },
    // toolbox: {
    //     feature: {
    //         restore: {},
    //         saveAsImage: {}
    //     }
    // },
    series: [{
        name: 'AQI',
        type: 'gauge',
        startAngle: 270,
        endAngle: -90,
        max: 500,
        center: [84,85],
        progress:{
            show: false
        },
        pointer: {
            show: true,
            width: 3,
            itemStyle: {
                color: '#8ec3e0',
            }
        },
        detail: {
            valueAnimation: true,
            formatter: ['{b|AQI}', '{value}'].join('\n'),
            fontSize: 22,
            padding: [-45,0,0,0],
            fontWeight: "bold",
            rich: {
                    b: {
                        color: '#444444',
                        fontSize: 25,
                        padding: [-10, 0, 10, 0],
                        fontWeight: "bold"
                    }
                }
        },
        data: [{
            value: this.draw_aqi_val,
            // name: 'IAQI'
        }],
        axisLine: {
            show: true,
            lineStyle: {
                color: [[1, new echarts.graphic.LinearGradient(0, 0.4, 0.7, 0.04, [{
                            offset: 0.1,
                            color: "#8fce00"
                        },
                        {
                            offset: 0.2,
                            color: "#ffd629"
                        },
                        {
                            offset: 0.3,
                            color: "#f6b26b"
                        },
                        {
                            offset: 0.4,
                            color: "#cc0000"
                        },
                        {
                            offset: 0.6,
                            color: "#a64d79"
                        },
                        {
                            offset: 1,
                            color: "#660000"
                        }
                    ])]],
                width: 10
            }
        },
        axisLabel: {
                distance: -6,
                show: true,
                formatter: function(value) {
                    // if (value === 0) {
                    //     return "Good"
                    // }
                    // if(value == 500){
                    //     return "Hazardous"
                    // }
                    // if(value == 100 || value == 200 || value == 300 || value == 400){
                    //   return value
                    // }
                },
                //   padding: '8 0 0 0'
                lineHeight: 12,
                fontSize:12,
                fontWeight: "bolder"
            },
            axisTick: {
            show: true,
            lineStyle: {
                color: 'auto',
                width: 0
            },
            length: -10
        },
        splitLine: {
            show: false,
            length: -10,
            distance: 0,
            lineStyle: {
                color: '#fff6db',
                width: 1
        },
        },
    }],

      });
    },
    draw_hour_humi () {
      let hourlyHumiChart = this.$echarts.init(document.getElementById('hourlyHumiChart'))
      hourlyHumiChart.setOption({
        series: [
    {
      type: 'gauge',
      startAngle: 90,
      endAngle: -270,
      max: 100,
      min: 0,
      clockwise: false,
      pointer: {
        show: false
      },
      progress: {
        show: true,
        overlap: false,
        roundCap: true,
        clip: false,
        itemStyle: {
          borderWidth: 1,
          borderColor: '#464646'
        }
      },
      axisLine: {
        lineStyle: {
          width: 20,
          // color: [1, '#b0e0e6'],
        }
      },
      splitLine: {
        show: false,
        distance: 0,
        length: 10
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        show: false,
        distance: 50
      },
      // data: gaugeData,
      data:[
      //   {
      //   name: "Humidity",
      //   title: {
      //     offsetCenter:['0%', '-60%']
      //   }
      // },
        {
    value: this.hourly_pred.Humi1,
    name: this.hourly_pred.Hour1,
    title: {
      offsetCenter: ['0%', '-65%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '-50%']
    }
  },
  {
    value: this.hourly_pred.Humi2,
    name: this.hourly_pred.Hour2,
    title: {
      offsetCenter: ['0%', '-28%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '-13%']
    }
  },
  {
    value: this.hourly_pred.Humi3,
    name: this.hourly_pred.Hour3,
    title: {
      offsetCenter: ['0%', '10%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '25%']
    }
  },
  {
    value: this.hourly_pred.Humi4,
    name: this.hourly_pred.Hour4,
    title: {
      offsetCenter: ['0%', '45%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '60%']
    }
  },
      ],
      title: {
        fontSize: 12,
        fontWeight:"Bold",
      },
      detail: {
        width: 25,
        height: 3,
        fontSize: 13,
        fontWeight: "Bold",
        color: 'inherit',
        borderColor: 'inherit',
        borderRadius: 25,
        borderWidth: 1,
        formatter: '{value} %'
      }
    }
  ],
  // grid: {
  //     x: 30,
  //     y: 20,
  //     backgroundColor: '#b0e0e6',
  //     borderColor: '#b0e0e6',
  //     show: true,
  //   }
      });
    },
    draw_hour_pres () {
      let hourlyPresChart = this.$echarts.init(document.getElementById('hourlyPresChart'))
      hourlyPresChart.setOption({
        series: [
    {
      type: 'gauge',
      startAngle: 90,
      endAngle: -270,
      max: 1060,
      min: 860,
      clockwise: false,
      pointer: {
        show: false
      },
      progress: {
        show: true,
        overlap: false,
        roundCap: true,
        clip: false,
        itemStyle: {
          borderWidth: 1,
          borderColor: '#464646'
        }
      },
      axisLine: {
        lineStyle: {
          width: 20,
          // color: [1, '#b0e0e6'],
        }
      },
      splitLine: {
        show: false,
        distance: 0,
        length: 10
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        show: false,
        distance: 50
      },
      // data: gaugeData,
      data:[
      //   {
      //   name: "Humidity",
      //   title: {
      //     offsetCenter:['0%', '-60%']
      //   }
      // },
        {
    value: this.hourly_pred.Pres1,
    name: this.hourly_pred.Hour1,
    title: {
      offsetCenter: ['0%', '-65%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '-50%']
    }
  },
  {
    value: this.hourly_pred.Pres2,
    name: this.hourly_pred.Hour2,
    title: {
      offsetCenter: ['0%', '-28%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '-13%']
    }
  },
  {
    value: this.hourly_pred.Pres3,
    name: this.hourly_pred.Hour3,
    title: {
      offsetCenter: ['0%', '10%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '25%']
    }
  },
  {
    value: this.hourly_pred.Pres4,
    name: this.hourly_pred.Hour4,
    title: {
      offsetCenter: ['0%', '45%']
    },
    detail: {
      valueAnimation: true,
      offsetCenter: ['0%', '60%']
    }
  },
      ],
      title: {
        fontSize: 12,
        fontWeight:"Bold",
      },
      detail: {
        width: 45,
        height: 3,
        fontSize: 13,
        fontWeight: "Bold",
        color: 'inherit',
        borderColor: 'inherit',
        borderRadius: 25,
        borderWidth: 1,
        formatter: '{value} hPa'
      }
    }
  ],
  // grid: {
  //     x: 30,
  //     y: 20,
  //     backgroundColor: '#b0e0e6',
  //     borderColor: '#b0e0e6',
  //     show: true,
  //   }
      });
    },
    getColorItem (p) {
      let mp = this.getPercentage
      for (let sub of this.linearColors) {
        if (!sub.ef && mp <= sub.v) {
          return sub
        } else if (sub.ef && mp < sub.v) {
          return sub
        }
      }
      return null
    },
    getColorItem2 (p) {
      let mp = p
      for (let sub of this.linearColors2) {
        if (!sub.ef && mp <= sub.v) {
          return sub
        } else if (sub.ef && mp < sub.v) {
          return sub
        }
      }
      return null
    }

  },
  props: {
    // 设置 进度条的 弧度
    bRadius: {
      type: Number,
      default: 4
    },
    textInside: {
      type: Number,
      default: 100
    },
    // 进度条的高度 就是粗细度
    strokeWidth: {
      type: Number,
      default: 8
    },
    strokeWidth2: {
      type: Number,
      default: 8
    },
    // 进度条 的百分比 [0-1] 的小数
    percentage: {
      type: Number,
      default: 0
    },
    // 进度条 每个阶段的 颜色组
    linearColors: {
      type: Array,
      default: function () {
        return [{ v: 25, s: '#00ff00', e: '#ffff00' }, { v: 50, s: '#00ff00', e: '#FFA500' }, { v: 66.67, s: '#00ff00', e: '#FFA500'}, { v: 91.66, s: '#00ff00', e: '#ff6f00' }, { v: 100, s: '#00ff00', e: '#FF0000', ef: false }]
      }
    },
    linearColors2: {
      type: Array,
      default: function () {
        return [{ v: 15, s: '#ecca00', e: '#ec9b00' }, { v: 30, s: '#ecca00', e: '#ec5300' }, { v: 45, s: '#ecca00', e: '#ec2400'}, { v: 60, s: '#ecca00', e: '#ec0000'}, { v: 75, s: '#ecca00', e: '#960018'}, { v: 100, s: '#ecca00', e: '#8d021f', ef: false }]
      }
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
body {
  max-width: 100%;
  margin: 0 auto;
}
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: black;
}
.icon, .iconfont {
  font-family:"iconfont" !important;
  font-size:20px;
  font-style:normal;
  -webkit-font-smoothing: antialiased;
  -webkit-text-stroke-width: 0.2px;
  -moz-osx-font-smoothing: grayscale;
}
span {
  font-weight: 200;
}

  .el-row {
    margin-bottom: 10px;
    &:last-child {
      margin-bottom: 0;
    }
  }
  .el-col {
    border-radius: 4px;
  }

  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
  .row-bg {
    padding: 10px 0;
    background-color: #f9fafc;
  }

  .br1{ line-height:30px}
  .br2{ line-height:15px}
  .br3{ line-height:20px}
  .br4{ line-height:40px}
  .br5{ line-height:32px}

  .el-header {
    background-color: #F2F6FC;
    color: #333;
    text-align: center;
    font-family:Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
    line-height: 40px;
  }

  .el-footer {
    background-color: #ffffff;
    color: #333;
    text-align: center;
    /* font-family:Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif; */
    /* line-height: 40px; */
  }
  
  .el-aside {
    background-color:white;
     /* #F2F6FC; */
    color: #333;
    text-align: center;
    line-height: 40px;
  }
  
  .el-main {
    background-color: #ffffff;
    color: #333;
    text-align: center;
    /* line-height: 30px; */
  }
  
  body > .el-container {
    margin-bottom: 40px;
  }

  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }
  
  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }

  .h1 {
  display: block;
  font-size: 2em;
  margin-top: 0.67em;
  margin-bottom: 0.67em;
  margin-left: 0;
  margin-right: 0;
  font-weight: bold;
  } 

  .text-wrapper {
  white-space: pre-wrap;
  }

  .p1 {
    word-break: break-all;
    width: 150px;
  }

  .el-progress{
  width:100%;		
  }
  .el-progress__text{
    color: #fff;
    font-size: 14px;
  }

  .progressbar >>> .el-progress-bar .el-progress-bar__outer {
    background-color: transparent;
  }

  .progressbar >>> .el-progress-bar .el-progress-bar__outer .el-progress-bar__inner {
    background-image: linear-gradient(to right, #00ff00 , #ffff00, #FFA500, #FF0000, #E6E6FA);
  }

  .progress{
  width: 500px;
  height: 20px;
  padding-left: 10px;
}

.progress ::v-deep .el-progress__text{
  color: #fff;
  font-size: 14px;
}
.progress ::v-deep .el-progress-bar__outer{
  height: 12px!important;
  border: 1px solid #78335f;
  background-color:transparent;
}

.progress ::v-deep .el-progress-bar__inner{
background-color: unset;
  background-image: linear-gradient(to right, #00ff00 , #ffff00, #FFA500, #FF0000, #E6E6FA);
}
  
.cp-progress-main {
  display: flex;
}
.cp-progress-main .cp-progress-bg {
    width: 50px;
    background: #eaedf4;
    /* background: #ff6f00; */
    flex: 1;
    margin: 8px 0;
}
.cp-progress-main .cp-progress-bg  .cp-progress-bar {
    transition: width 1s;
}


</style>
