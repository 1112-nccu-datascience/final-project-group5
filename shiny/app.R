library(shiny)
# library(ggbiplot)
library(ggplot2)
library(DT)
library(FactoMineR)

# Define UI for application that draws a histogram
ui <- fluidPage(
  #tags$style(HTML('body {font-family:"Lucida Sans Unicode", sans-serif; font-weight:bold;}')),
  tags$head(
    tags$style(HTML("
    .navbar-nav > li > a, .navbar-brand {
                   padding-top:10px !important; 
                   padding-bottom:10px !important;
                   height: 40px;
                 }
                 .navbar {min-height:25px !important;}

      body {
        background-color: #EEF1EA;
      }
      
      h2 {
        font-family: Lucida Sans Unicode, 
        sans-serif;
        color:#798233;
        font-size: 16px;
        font-weight: bold
      }
             .shiny-input-container {
        color: #403B3B;
      }
      #sidebar {
             background-color: #D2E8EE;
             }
             body, label, input, button, select {
             color: #856A6A;
             font-family: Lucida Sans Unicode, sans-serif;
             font-weight: bold;
             font-size: 15px;}
                  
       
                    "))
     
  ),
  
  # Application title

  navbarPage(p("Group 5 - Final Project", style = "color:#8F9650;font-size: 22px;font-family: Lucida Sans Unicode, sans-serif;font-weight: bold"),
             tabPanel(p("Dataset", style = "color:#798233;font-size: 17px;font-family: Lucida Sans Unicode, sans-serif;font-weight: bold"), 
                      tabsetPanel(
                        tabPanel( h2('Data'), DT::dataTableOutput('data') 
                        ),
                        tabPanel( h2('summary'), verbatimTextOutput('data_summary') 
                        )
                      )),
             
             tabPanel(p("EDA", style = "color:#798233;font-size: 17px;font-family: Lucida Sans Unicode, sans-serif;font-weight: bold"), 
                      tabsetPanel(
                        tabPanel(h2('Target Count'), imageOutput('target_count')
                        ),
                        tabPanel(h2('Missing Values'),imageOutput('NA_count')
                        ),
                        tabPanel(h2('Feature Correlation - With NA'), imageOutput('corre_NA')
                        ),
                        tabPanel(h2('Feature Correlation - Without NA'), imageOutput('corre')
                        ) 
                      )),
             tabPanel(p("PCA", style = "color:#798233;font-size: 17px;font-family: Lucida Sans Unicode, sans-serif;font-weight: bold"),
                        tabsetPanel(
                          tabPanel(h2('PCA'), 
                                   br(),
                                   div("[ PCA ]", style = "text-align: center;  
                                     color:#B2BA81; font-size:120%; width: 80%;"),
                                   imageOutput("pca_plot",width = "90vw", height = "90vh")
                                   ),
                        )
                      ),
             
             tabPanel(p("Results", style = "color:#798233;font-size: 17px;font-family: Lucida Sans Unicode, sans-serif;font-weight: bold"), 
                      tabsetPanel(
                        tabPanel(h2('XGBoost'), 
                                 div("[ XGBoost Model Training = 0.266 ]", style = "text-align: center;  
                                     color:#798233; font-size:120%; width: 60%;"),
                                 imageOutput("xgboost",width = "100vw", height = "100vh"),
                                 
                                 div("[ XGBoost Model Training - PCA = 0.283 ]", style = "text-align: center;
                                     color:#798233; font-size:120%; width: 60%;"),
                                 imageOutput("xgboost_PCA",width = "100vw", height = "100vh"),
                                 
                                 ),
                        tabPanel(h2('Results'), 
                                 br(),
                                 div("[ Comparison Results ]", style = "text-align: center;  
                                     color:#798233; font-size:120%; width: 60%;"),
                                 plotOutput("all_result"),
                        )
                      ))
            ),
)

# Define server logic required to draw a histogram
server <- function(input, output,session) {

  df <- read.csv("./subset_file2.csv", header = T, sep = "," , row.names = 1)
  train <- read.csv("./train.csv", header = T, sep = "," , row.names = 1)
  result <- read.csv("./result.csv", header = T, sep = "," )
  
  
  df[df == -1] <- NA
  output$data <- renderDT({df}, options = list(pageLength = 25) )
  output$data_summary <- renderPrint(summary(train))
  output$target_count <- renderImage({
                          list(src = "./TargetCount.png",
                               alt = "Image",
                               width = "70%",
                               height = "auto",
                               preserveAspectRatio = TRUE)
                        }, deleteFile = FALSE)
  output$NA_count <- renderImage({
    list(src = "./CountMissingValues.png",
         alt = "Image",
         width = "70%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)

  output$corre_NA <- renderImage({
    list(src = "./FeaturesCorrelation-WithNA.png",
         alt = "Image",
         width = "70%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)
  
  output$corre <- renderImage({
    list(src = "./FeaturesCorrelation-WithoutNA.png",
         alt = "Image",
         width = "70%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)
  
  output$pca_plot <- renderImage({
    list(src = "./PCA.png",
         alt = "Image",
         width = "100%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)
  
  output$xgboost <- renderImage({
    list(src = "./XGBoost_NoPCA.png",
         alt = "Image",
         width = "60%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)
  
  output$xgboost_PCA <- renderImage({
    list(src = "./XGBoost_PCA.png",
         alt = "Image",
         width = "60%",
         height = "auto",
         preserveAspectRatio = TRUE)
  }, deleteFile = FALSE)
  
  # output$nb <- renderImage({
  #   list(src = "./nb.png",
  #        alt = "Image",
  #        width = "55%",
  #        height = "auto",
  #        preserveAspectRatio = TRUE)
  # }, deleteFile = FALSE)
  # 
  # output$nb_pca <- renderImage({
  #   list(src = "./nb_pca.png",
  #        alt = "Image",
  #        width = "55%",
  #        height = "auto",
  #        preserveAspectRatio = TRUE)
  # }, deleteFile = FALSE)
  # 
  # output$lr <- renderImage({
  #   list(src = "./lr.png",
  #        alt = "Image",
  #        width = "55%",
  #        height = "auto",
  #        preserveAspectRatio = TRUE)
  # }, deleteFile = FALSE)
  # 
  # output$lr_pca <- renderImage({
  #   list(src = "./lr_pca.png",
  #        alt = "Image",
  #        width = "55%",
  #        height = "auto",
  #        preserveAspectRatio = TRUE)
  # }, deleteFile = FALSE)
  
  output$all_result <- renderPlot({
    g <- ggplot( data = result, aes(x = Model, y = NormalGini)) 
      g <- g + geom_bar(stat = "identity", width = 0.5 , fill='#F4C7B0') 
      g <- g + labs(x = "Model", y = "Normal Gini", title = "Comparison Results") 
      g <- g + theme_minimal() 
      # g <- g + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(g)
  })
    
}


# Run the application 
shinyApp(ui = ui, server = server)