# Functions for the research page

# Load inside libraries
library(tidyverse)
library(stringi)

# Variables
MONTHS <- c(
  "dec", "nov", "oct", "sep", "aug", "jul",
  "jun", "may", "apr", "mar", "feb", "jan"
)

# Format authors
format_author <- function(authors) {
  authors_modif <- stri_replace_all(authors, regex = " and", replacement = ",")
  stri_replace_last(authors_modif, replacement = " and", regex = ",")
}

# Box publication function
box_publication <- function(data) {
  tagList(
    tags$span(
      class = "text-primary",
      "> ",
      data$title[1]
    ),
    tags$span(
      class = "text-primary",
      "-",
      tags$i(
        data$journal[1]
      )
    ),
    div(
      class = "d-flex justify-content-between mt-2",
      format_author(data$author)
    ),
    div(
      class = "d-flex justify-content-between mt-2",
      div(
        div(
          tags$a(
            href = data$url[1],
            role = "button",
            fa("globe")
          ),
          tags$a(
            href = data$arxiv_url[1],
            role = "button",
            tags$i(
              class='ai ai-arxiv'
            )
          )
        )
      ),
      div(
        style="align: right;",
        paste0(str_to_title(data$month), ". ", data$year)
      )
    )
  )
}


# Box preprint function
box_preprint <- function(data) {
  tagList(
    tags$span(
      class = "text-primary",
      "> ",
      data$title[1]
    ),
    div(
      class = "d-flex justify-content-between mt-2",
      format_author(data$author)
    ),
    div(
      class = "d-flex justify-content-between mt-2",
      div(
        div(
          tags$a(
            href = data$arxiv_url[1],
            role = "button",
            tags$i(
              class='ai ai-arxiv'
            )
          )
        )
      ),
      div(
        style="align: right;",
        paste0(str_to_title(data$month), ". ", data$year)
      )
    )
  )
}

# Select publication
selected_year_item <- function(data, selected_year, id, collapse) {
  data_selected_year <- data[data$year == selected_year, ]
  tag_year <- NULL

  data_year  <- data_selected_year |> arrange(factor(month, levels = MONTHS))
  for (i in 1:dim(data_year)[1]) {
    if (data_year[i, ]$journal == "") {
      div_article <- box_preprint(data_year[i, ])
    } else {
      div_article <- box_publication(data_year[i, ])
    }

    tag <- div(
      id = sprintf(
        "%s-collapse%s",
        id,
        data_selected_year$year[i]
      ),
      class = sprintf(
        "accordion-collapse collapse %s",
        collapse
      ),
      "aria-labelledby" = sprintf(
        "%s-heading%s",
        id,
        data_selected_year$year[i]
      ),
      div(
        class = "accordion-body border-top border-primary",
        div_article
      )
    )
    tag_year <- tagList(tag_year, tag)
  }
  
  tag_year
}


# Accordion function
accordeon_mult_code <- function(data, id, show = TRUE) {
  if (show) {
    collapse <- "show"
    button_collapse <- ""
  } else {
    collapse <- "hide"
    button_collapse <- "collapsed"
  }

  final_tag <- NULL
  unique_year <- unique(data$year)

  for (selected_year in sort(unique_year, decreasing = TRUE)) {
    tag <- div(
      class = "g-col-12 align-self-center g-start-1",
      div(
        class = paste(
          "accordion accordion-flush bg-white text-primary",
          "border-primary border border-1 accordion-flush"
        ),
        id = sprintf("accordeon-adapt-%s", id),
        div(
          class = "accordion-item",
          div(
            id = sprintf("%s-heading%s", id, selected_year),
            tags$button(
              class = sprintf(
                "accordion-button bg-white text-primary p-2 %s",
                button_collapse
              ),
              type = "button",
              "data-bs-toggle" = "collapse",
              "data-bs-target" = sprintf("#%s-collapse%s", id, selected_year),
              "aria-expanded" = "true",
              "aria-controls" = sprintf("%s-collapse%s", id, selected_year),
              div(class = "fw-bold text-primary", selected_year)
            ),
            selected_year_item(
              data = data,
              selected_year = selected_year,
              id = id,
              collapse = collapse
            )
          )
        )
      )
    )
    final_tag <- tagList(final_tag, tag)
  }

  final_tag
}
