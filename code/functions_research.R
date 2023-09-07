#' Create an accordion element
#'
#' This function generates an accordion element using HTML markup and the
#' Bootstrap classes. The function creates a button element with the given
#' \code{title} as the button text, which when clicked, toggles the visibility
#' of the \code{body} element. The \code{id} argument is used to ensure that
#' the button and collapsible content are associated correctly.
#'
#' @param title character string representing the title of accordion element.
#' @param body the content to be displayed when accordion element is expanded.
#' @param id character string representing a unique id for accordion element.
#' @param show logical value whether accordion element is initially expanded.
#'
#' @return an HTML element representing the accordion element.
#'
#' @examples
#' library(htmltools)
#' accordeon_code(
#'   title = "Accordion Element",
#'   body = "This is the content that will be displayed.",
#'   id = "example-accordion"
#' )
#' @importFrom htmltools div tags
#' @export
accordeon_code <- function(title, body, id, show=TRUE) {
  if (show == TRUE) {
    collapse <- "show"
    button_collapse <- ""
  } else {
    collapse <- "hide"
    button_collapse <- "collapsed"
  }
  div(  
    class = "accordion bg-white text-primary border-primary border border-1 accordion-flush",
    id = sprintf("accordeon-adapt-%s", id),
    div(
      class = "accordion-item",
      div(
        id = sprintf("%s-headingOne", id),
        tags$button(
            class = sprintf(
                "accordion-button bg-white text-primary p-2 %s", button_collapse
            ),
            type = "button",
            "data-bs-toggle" = "collapse",
            "data-bs-target" = sprintf("#%s-collapseOne", id),
            "aria-expanded" = "true",
            "aria-controls" = sprintf("%s-collapseOne", id),
            div(
                class = "fw-bold text-primary",
                title
            )
        ),
        div(
          id = sprintf("%s-collapseOne", id),
          class = sprintf("accordion-collapse collapse %s", collapse),
          "aria-labelledby" = sprintf("%s-headingOne", id),
          div(
            class = "accordion-body border-top border-primary",
            body
          )
        )
      )
    )
  )
}


# Box function
box_publication <- function(data) {
  tagList(
    tags$span(
      class = "text-primary",
      "> ",
      data$title[1]
    ),
    tags$span(
      class = "text-primary",
      data$journal[1]
    ),
    div(
      class = "d-flex justify-content-between mt-2",
      div(
        class = "text-primary",
        data$author,
        tags$a(
          href = data$url[1],
          role = "button",
          fa("globe")
        )
      )
    )
  )
}


# Select publication
selected_year_item <- function(data, selected_year, id, collapse) {
  data_selected_year <- data[data$year == selected_year, ]
  tag_year <- NULL

  for (i in 1:dim(data_selected_year)[1]) {
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
        box_publication(data_selected_year[i, ])
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
