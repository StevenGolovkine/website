# Functions for the about page

# Accordion function
accordeon_code <- function(title, body, id, show = TRUE) {
    if (show == TRUE) {
        collapse <- "show"
        button_collapse <- ""
    } else {
        collapse <- "hide"
        button_collapse <- "collapsed"
    }

    div(
        class = paste(
            "accordion bg-white text-primary border-primary border",
            "border-1 accordion-flush"
        ),
        id = sprintf("accordeon-adapt-%s", id),
        div(
            class = "accordion-item",
            div(
                id = sprintf("%s-headingOne", id),
                tags$button(
                    class = sprintf(
                        "accordion-button bg-white text-primary p-2 %s",
                        button_collapse
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

# Format resume function
resume_code <- function(title, subtitle, text = NULL, place, date) {
    div(
        class = "grid",
        style = "--bs-gap: 0rem;",
        div(
            class = "g-col-12 g-col-md-8 align-self-start g-start-1",
            div(
                p(class = "m-1 fw-bold", title),
                p(class = "m-1", subtitle),
                if (is.null(subtitle) == FALSE) {
                    p(class = "m-1", text)
                }
            )
        ),
        div(
            class = "g-col-12 g-col-md-4 align-self-end g-start-1 g-start-md-9",
            div(
                p(class = "m-1 text-start text-md-end", place),
                p(class = "m-1 text-start text-md-end", date)
            )
        )
    )
}
